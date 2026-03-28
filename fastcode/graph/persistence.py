"""Graph persistence — save and load KnowledgeGraph and AnalysisMeta.

File layout under a project root:
    .fastcode/knowledge-graph.json   — main graph
    .fastcode/meta.json              — AnalysisMeta
    .fastcode/issues.json            — GraphIssue list (optional)

Design: load is lenient (normalize then parse), save is strict (validate first).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AnalysisMeta, GraphIssue, KnowledgeGraph
from .validation import normalize_graph_data, strip_invalid_edges, validate_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASTCODE_DIR = ".fastcode"
GRAPH_FILE = "knowledge-graph.json"
META_FILE = "meta.json"
ISSUES_FILE = "issues.json"


def _fastcode_dir(project_root: Path) -> Path:
    return project_root / FASTCODE_DIR


# ---------------------------------------------------------------------------
# save_graph
# ---------------------------------------------------------------------------


def save_graph(
    graph: KnowledgeGraph,
    project_root: Path,
    *,
    strict: bool = True,
    issues: list[GraphIssue] | None = None,
) -> None:
    """Persist a KnowledgeGraph to .fastcode/knowledge-graph.json.

    Args:
        graph: The graph to save.
        project_root: Root of the project being analysed.
        strict: If True, raise on validation errors before saving.
        issues: Optional list of issues to co-save in issues.json.
    """
    if strict:
        result = validate_graph(graph)
        if not result.valid:
            msgs = [i.message for i in result.issues if i.severity in ("error", "fatal")]
            raise ValueError(f"Graph failed validation before save: {msgs}")

    d = _fastcode_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)

    graph_path = d / GRAPH_FILE
    graph_path.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if issues is not None:
        issues_path = d / ISSUES_FILE
        issues_path.write_text(
            json.dumps([i.model_dump() for i in issues], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# load_graph
# ---------------------------------------------------------------------------


def load_graph(project_root: Path, *, auto_fix: bool = True) -> KnowledgeGraph:
    """Load a KnowledgeGraph from .fastcode/knowledge-graph.json.

    Applies normalize_graph_data before parsing (lenient load).
    If auto_fix is True, strips edges referencing missing nodes.
    """
    graph_path = _fastcode_dir(project_root) / GRAPH_FILE
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph file found at {graph_path}")

    data: dict[str, Any] = json.loads(graph_path.read_text(encoding="utf-8"))
    data = normalize_graph_data(data)
    graph = KnowledgeGraph.model_validate(data)

    if auto_fix:
        graph, _ = strip_invalid_edges(graph)

    return graph


# ---------------------------------------------------------------------------
# save_meta / load_meta
# ---------------------------------------------------------------------------


def save_meta(meta: AnalysisMeta, project_root: Path) -> None:
    """Persist AnalysisMeta to .fastcode/meta.json."""
    d = _fastcode_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / META_FILE
    meta_path.write_text(
        json.dumps(meta.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_meta(project_root: Path) -> AnalysisMeta:
    """Load AnalysisMeta from .fastcode/meta.json."""
    meta_path = _fastcode_dir(project_root) / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta file found at {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return AnalysisMeta.model_validate(data)


# ---------------------------------------------------------------------------
# load_issues
# ---------------------------------------------------------------------------


def load_issues(project_root: Path) -> list[GraphIssue]:
    """Load issues list from .fastcode/issues.json. Returns empty list if absent."""
    issues_path = _fastcode_dir(project_root) / ISSUES_FILE
    if not issues_path.exists():
        return []
    data = json.loads(issues_path.read_text(encoding="utf-8"))
    return [GraphIssue.model_validate(i) for i in data]


# ---------------------------------------------------------------------------
# graph_exists
# ---------------------------------------------------------------------------


def graph_exists(project_root: Path) -> bool:
    """Return True if a persisted graph file exists."""
    return (_fastcode_dir(project_root) / GRAPH_FILE).exists()
