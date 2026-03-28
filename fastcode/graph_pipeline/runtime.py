"""Runtime orchestrator — full graph build pipeline.

Entry point: build_graph()

Pipeline order:
    Scanner -> StructuralAnalyzer -> Assembler -> PipelineValidator -> save_graph
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from fastcode.graph.models import AnalysisMeta, KnowledgeGraph, ProjectMeta
from fastcode.graph.persistence import save_graph, save_meta
from fastcode.graph.validation import strip_invalid_edges, validate_graph
from fastcode.symbol_backend.protocol import SymbolProvider

from .assembler import Assembler
from .scanner import Scanner
from .structural_analyzer import StructuralAnalyzer
from .validator import validate_pipeline_result

logger = logging.getLogger(__name__)

_GRAPH_VERSION = "2.0"
_BACKEND_VERSION = "0.2.0"


def build_graph(
    project_root: Path,
    provider: SymbolProvider,
    project_meta: ProjectMeta,
    *,
    exclude_patterns: list[str] | None = None,
    include_extensions: list[str] | None = None,
    strict_save: bool = False,
) -> KnowledgeGraph:
    """Run a full graph build for *project_root*.

    Args:
        project_root:       Root directory of the project to analyse.
        provider:           Symbol backend to use for file analysis.
        project_meta:       Project-level metadata (name, languages, …).
        exclude_patterns:   Override default file exclusion patterns.
        include_extensions: Override default extension allowlist.
        strict_save:        If True, raise on validation errors before saving.

    Returns:
        The assembled and persisted KnowledgeGraph.
    """
    project_root = project_root.resolve()
    logger.info("build_graph: starting full build for %s", project_root)

    # 1. Scan
    scanner = Scanner(
        exclude_patterns=exclude_patterns,
        include_extensions=include_extensions,
    )
    scan_result = scanner.scan(project_root)

    # 2. Structural analysis
    analyzer = StructuralAnalyzer()
    analysis_result = analyzer.analyze(scan_result, provider)

    # 3. Assemble graph
    assembler = Assembler()
    assembly = assembler.assemble(analysis_result.symbol_index, scan_result)

    # 4. Build KnowledgeGraph object
    graph = KnowledgeGraph(
        version=_GRAPH_VERSION,
        project=project_meta,
        nodes=assembly.nodes,
        edges=assembly.edges,
    )

    # 5. Strip structurally invalid edges (dangling refs)
    graph, stripped_issues = strip_invalid_edges(graph)
    if stripped_issues:
        logger.info("build_graph: stripped %d invalid edge(s)", len(stripped_issues))

    # 6. Pipeline validation
    pipeline_val = validate_pipeline_result(graph)
    if not pipeline_val.valid:
        for issue in pipeline_val.issues:
            logger.warning("Pipeline issue [%s]: %s", issue.severity, issue.message)

    # 7. Graph structural validation
    graph_val = validate_graph(graph)
    issues_to_save = pipeline_val.issues + graph_val.issues

    # 8. Persist
    save_graph(graph, project_root, strict=strict_save, issues=issues_to_save)

    # 9. Persist AnalysisMeta
    meta = AnalysisMeta(
        graph_version=_GRAPH_VERSION,
        backend_version=_BACKEND_VERSION,
        last_analyzed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit_hash="",
        analyzed_files=analysis_result.analyzed_files,
        analysis_mode="full",
        symbol_backend=provider.backend_name,  # type: ignore[arg-type]
        serena_available=False,
        runtime_mode="restricted",
        warnings_count=sum(1 for i in issues_to_save if i.severity == "warning"),
    )
    save_meta(meta, project_root)

    logger.info(
        "build_graph: done — %d nodes, %d edges, saved to %s/.fastcode/",
        len(graph.nodes),
        len(graph.edges),
        project_root,
    )
    return graph
