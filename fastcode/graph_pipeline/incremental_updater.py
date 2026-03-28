"""Incremental graph updater.

Only re-analyses changed files and splices the results into the existing
persisted graph. Nodes and edges for unchanged files are kept as-is.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from fastcode.graph.models import AnalysisMeta, GraphEdge, GraphNode, KnowledgeGraph
from fastcode.graph.persistence import (
    graph_exists,
    load_graph,
    load_meta,
    save_graph,
    save_meta,
)
from fastcode.graph.validation import strip_invalid_edges
from fastcode.symbol_backend.protocol import SymbolProvider

from .assembler import Assembler
from .validator import validate_pipeline_result

logger = logging.getLogger(__name__)


def incremental_update(
    project_root: Path,
    changed_files: list[Path],
    provider: SymbolProvider,
    *,
    strict_save: bool = False,
) -> KnowledgeGraph:
    """Re-analyse *changed_files* and merge results into the existing graph.

    Args:
        project_root:  Root of the project (must have an existing graph).
        changed_files: Files whose on-disk content has changed.
        provider:      Symbol backend to use for re-analysis.
        strict_save:   If True, raise on validation errors before saving.

    Returns:
        The updated KnowledgeGraph.

    Raises:
        FileNotFoundError: If no existing graph is found under project_root.
    """
    project_root = project_root.resolve()

    if not graph_exists(project_root):
        raise FileNotFoundError(
            f"No existing graph found under {project_root}/.fastcode/. "
            "Run a full build first."
        )

    graph = load_graph(project_root)
    changed_paths = {str(fp.resolve()) for fp in changed_files}

    # ------------------------------------------------------------------
    # 1. Remove nodes + edges that belong to changed files
    # ------------------------------------------------------------------
    surviving_nodes: list[GraphNode] = [
        n for n in graph.nodes
        if n.file_path not in changed_paths
    ]
    surviving_node_ids = {n.id for n in surviving_nodes}
    surviving_edges: list[GraphEdge] = [
        e for e in graph.edges
        if e.source in surviving_node_ids and e.target in surviving_node_ids
    ]

    logger.info(
        "incremental_update: removed %d node(s) and %d edge(s) for %d changed file(s)",
        len(graph.nodes) - len(surviving_nodes),
        len(graph.edges) - len(surviving_edges),
        len(changed_files),
    )

    # ------------------------------------------------------------------
    # 2. Re-analyse changed files
    # ------------------------------------------------------------------
    from fastcode.symbol_backend.symbol_index import SymbolIndex

    index = SymbolIndex()
    failed: list[str] = []

    for fp in changed_files:
        fp = fp.resolve()
        if not fp.is_file():
            logger.warning("incremental_update: skipping missing file %s", fp)
            failed.append(str(fp))
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            symbols, relationships = provider.analyze_file(fp, content)
            index.add(symbols, relationships)
        except Exception as exc:  # noqa: BLE001
            logger.warning("incremental_update: provider failed on %s: %s", fp, exc)
            failed.append(str(fp))

    # ------------------------------------------------------------------
    # 3. Assemble new nodes/edges from re-analysed files
    # ------------------------------------------------------------------
    assembler = Assembler()
    assembly = assembler.assemble(index)

    # ------------------------------------------------------------------
    # 4. Merge into graph
    # ------------------------------------------------------------------
    graph.nodes = surviving_nodes + assembly.nodes
    graph.edges = surviving_edges + assembly.edges

    graph, _ = strip_invalid_edges(graph)

    # ------------------------------------------------------------------
    # 5. Validate & persist
    # ------------------------------------------------------------------
    pipeline_val = validate_pipeline_result(graph)
    issues = pipeline_val.issues

    save_graph(graph, project_root, strict=strict_save, issues=issues)

    # Update meta
    try:
        old_meta = load_meta(project_root)
        symbol_backend = old_meta.symbol_backend
        runtime_mode = old_meta.runtime_mode
        serena_available = old_meta.serena_available
    except FileNotFoundError:
        symbol_backend = "ast"  # type: ignore[assignment]
        runtime_mode = "restricted"  # type: ignore[assignment]
        serena_available = False

    meta = AnalysisMeta(
        graph_version="2.0",
        backend_version="0.2.0",
        last_analyzed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit_hash="",
        analyzed_files=len(changed_files),
        changed_files=[str(fp) for fp in changed_files],
        analysis_mode="incremental",
        symbol_backend=symbol_backend,
        serena_available=serena_available,
        runtime_mode=runtime_mode,
        warnings_count=sum(1 for i in issues if i.severity == "warning"),
    )
    save_meta(meta, project_root)

    logger.info(
        "incremental_update: done — graph now has %d nodes, %d edges",
        len(graph.nodes),
        len(graph.edges),
    )
    return graph
