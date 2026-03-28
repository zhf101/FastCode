"""Graph validation and normalize logic.

Responsibilities:
- validate_graph(): structural correctness check on a KnowledgeGraph object
- normalize_graph_data(): lenient in-place normalize on raw dict before parsing

NOT responsible for pipeline build-result completeness (that lives in
graph_pipeline/validator.py in a later phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import (
    EdgeType,
    GraphEdge,
    GraphIssue,
    GraphNode,
    KnowledgeGraph,
    NodeType,
)

# ---------------------------------------------------------------------------
# Alias normalization tables
# ---------------------------------------------------------------------------

_NODE_TYPE_ALIASES: dict[str, NodeType] = {
    "File": "file",
    "Function": "function",
    "method": "function",
    "Method": "function",
    "Class": "class",
    "Interface": "class",
    "Module": "module",
    "package": "module",
    "Package": "module",
    "Concept": "concept",
}

_EDGE_TYPE_ALIASES: dict[str, EdgeType] = {
    "Contains": "contains",
    "Imports": "imports",
    "import": "imports",
    "Calls": "calls",
    "call": "calls",
    "Inherits": "inherits",
    "extends": "inherits",
    "Extends": "inherits",
    "DependsOn": "depends_on",
    "depends": "depends_on",
    "Related": "related",
    "similar": "related",
}

_COMPLEXITY_ALIASES: dict[str, str] = {
    "Low": "low",
    "Moderate": "moderate",
    "medium": "moderate",
    "Medium": "moderate",
    "High": "high",
}

_DIRECTION_ALIASES: dict[str, str] = {
    "Directed": "directed",
    "Undirected": "undirected",
    "bidirectional": "undirected",
}


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    valid: bool
    issues: list[GraphIssue] = field(default_factory=list)
    fatal: bool = False

    def add(self, issue: GraphIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "fatal":
            self.fatal = True
            self.valid = False
        elif issue.severity == "error":
            self.valid = False


# ---------------------------------------------------------------------------
# normalize_graph_data — lenient, operates on raw dict
# ---------------------------------------------------------------------------


def normalize_graph_data(data: dict[str, Any]) -> dict[str, Any]:
    """Apply in-place normalizations on raw dict before Pydantic parsing.

    Returns the (mutated) dict for convenience.
    """
    # Normalize nodes
    for node in data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        # type aliases
        if node.get("type") in _NODE_TYPE_ALIASES:
            node["type"] = _NODE_TYPE_ALIASES[node["type"]]
        # summary default
        if not node.get("summary"):
            node["summary"] = node.get("name", "")
        # tags default
        if node.get("tags") is None:
            node["tags"] = []
        # complexity aliases
        if node.get("complexity") in _COMPLEXITY_ALIASES:
            node["complexity"] = _COMPLEXITY_ALIASES[node["complexity"]]
        # source default
        if not node.get("source"):
            node["source"] = "static"

    # Normalize edges
    valid_edges: list[dict] = []
    for edge in data.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if not edge.get("source") or not edge.get("target"):
            continue  # discard
        # type aliases
        if edge.get("type") in _EDGE_TYPE_ALIASES:
            edge["type"] = _EDGE_TYPE_ALIASES[edge["type"]]
        # weight default
        if edge.get("weight") is None:
            edge["weight"] = 0.5
        # direction aliases
        if edge.get("direction") in _DIRECTION_ALIASES:
            edge["direction"] = _DIRECTION_ALIASES[edge["direction"]]
        valid_edges.append(edge)
    data["edges"] = valid_edges

    return data


# ---------------------------------------------------------------------------
# validate_graph — strict, operates on parsed KnowledgeGraph
# ---------------------------------------------------------------------------


def validate_graph(graph: KnowledgeGraph) -> ValidationResult:
    """Validate structural integrity of a parsed KnowledgeGraph.

    Checks:
    - project exists (fatal if missing — caught by Pydantic already)
    - node IDs are unique
    - edge source/target reference existing nodes
    - layer node_ids reference existing nodes
    - tour node_ids reference existing nodes
    - nodes list is non-empty after parsing (fatal)
    """
    result = ValidationResult(valid=True)

    # Fatal: no nodes
    if not graph.nodes:
        result.add(
            GraphIssue(
                severity="fatal",
                message="Graph has no nodes after parsing",
            )
        )
        return result

    # Node ID uniqueness
    node_ids: set[str] = set()
    for node in graph.nodes:
        if node.id in node_ids:
            result.add(
                GraphIssue(
                    severity="error",
                    message=f"Duplicate node id: '{node.id}'",
                    node_id=node.id,
                )
            )
        node_ids.add(node.id)

    # Edge referential integrity
    for edge in graph.edges:
        if edge.source not in node_ids:
            result.add(
                GraphIssue(
                    severity="error",
                    message=f"Edge source '{edge.source}' references unknown node",
                    edge_source=edge.source,
                    edge_target=edge.target,
                )
            )
        if edge.target not in node_ids:
            result.add(
                GraphIssue(
                    severity="error",
                    message=f"Edge target '{edge.target}' references unknown node",
                    edge_source=edge.source,
                    edge_target=edge.target,
                )
            )

    # Layer node_ids
    for layer in graph.layers:
        for nid in layer.node_ids:
            if nid not in node_ids:
                result.add(
                    GraphIssue(
                        severity="warning",
                        message=f"Layer '{layer.id}' references unknown node '{nid}'",
                    )
                )

    # Tour node_ids
    for step in graph.tour:
        for nid in step.node_ids:
            if nid not in node_ids:
                result.add(
                    GraphIssue(
                        severity="warning",
                        message=f"TourStep order={step.order} references unknown node '{nid}'",
                    )
                )

    return result


# ---------------------------------------------------------------------------
# strip_invalid_edges — auto-fix helper used by persistence on load
# ---------------------------------------------------------------------------


def strip_invalid_edges(graph: KnowledgeGraph) -> tuple[KnowledgeGraph, list[GraphIssue]]:
    """Remove edges that reference non-existent nodes, return cleaned graph + issues."""
    node_ids = {n.id for n in graph.nodes}
    good: list[GraphEdge] = []
    issues: list[GraphIssue] = []
    for edge in graph.edges:
        if edge.source in node_ids and edge.target in node_ids:
            good.append(edge)
        else:
            issues.append(
                GraphIssue(
                    severity="warning",
                    message=(
                        f"Stripped edge {edge.source} -> {edge.target} ({edge.type}): "
                        "references unknown node"
                    ),
                    edge_source=edge.source,
                    edge_target=edge.target,
                    auto_fixed=True,
                )
            )
    graph = graph.model_copy(update={"edges": good})
    return graph, issues
