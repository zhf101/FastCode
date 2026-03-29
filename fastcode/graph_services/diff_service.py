"""Diff service — impact analysis for changed files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastcode.graph.models import GraphEdge, GraphNode, KnowledgeGraph, Layer
from fastcode.graph import search


@dataclass
class DiffContext:
    changed_files: list[str]
    changed_nodes: list[GraphNode]
    affected_nodes: list[GraphNode]   # bounded transitive neighbours of changed nodes
    impacted_edges: list[GraphEdge]
    affected_layers: list[Layer]
    unmapped_files: list[str]         # changed files with no matching node
    risk_summary: list[str]


def build_diff_context(
    graph: KnowledgeGraph,
    changed_files: list[str],
    *,
    max_hops: int = 2,
) -> DiffContext:
    """Compute impact context for *changed_files*."""
    changed_nodes: list[GraphNode] = []
    unmapped: list[str] = []

    for fp in changed_files:
        matches = _match_changed_file_nodes(graph, fp)
        if matches:
            changed_nodes.extend(matches)
        else:
            unmapped.append(fp)

    changed_ids = {n.id for n in changed_nodes}
    affected_ids, impacted_edges, max_impact_depth = _collect_impacted_neighbourhood(
        graph,
        changed_ids,
        max_hops=max_hops,
    )

    affected_nodes = [n for n in graph.nodes if n.id in affected_ids]

    # Affected layers
    all_involved = changed_ids | affected_ids
    affected_layers = [
        l for l in graph.layers
        if any(nid in l.node_ids for nid in all_involved)
    ]

    # Risk summary
    risk: list[str] = []
    if len(affected_layers) > 1:
        risk.append(f"Change spans {len(affected_layers)} layers: " +
                    ", ".join(l.name for l in affected_layers))
    complex_changed = [
        n for n in changed_nodes if n.complexity == "high"
    ]
    if complex_changed:
        risk.append(f"{len(complex_changed)} high-complexity node(s) changed: " +
                    ", ".join(n.name for n in complex_changed))
    if len(affected_nodes) > 5:
        risk.append(f"Large downstream impact: {len(affected_nodes)} affected node(s)")
    if max_impact_depth > 1:
        risk.append(f"Transitive impact reaches depth {max_impact_depth}")
    if unmapped:
        risk.append(f"{len(unmapped)} changed file(s) not in graph: " + ", ".join(unmapped))

    return DiffContext(
        changed_files=changed_files,
        changed_nodes=changed_nodes,
        affected_nodes=affected_nodes,
        impacted_edges=list({(e.source, e.target): e for e in impacted_edges}.values()),
        affected_layers=affected_layers,
        unmapped_files=unmapped,
        risk_summary=risk,
    )


def _collect_impacted_neighbourhood(
    graph: KnowledgeGraph,
    changed_ids: set[str],
    *,
    max_hops: int,
) -> tuple[set[str], list[GraphEdge], int]:
    """Collect a bounded transitive neighbourhood around changed nodes."""
    if not changed_ids:
        return set(), [], 0

    affected_ids: set[str] = set()
    impacted_edges: dict[tuple[str, str, str], GraphEdge] = {}
    frontier = set(changed_ids)
    visited = set(changed_ids)
    max_depth_reached = 0

    for depth in range(1, max(1, max_hops) + 1):
        next_frontier: set[str] = set()
        for nid in frontier:
            for edge in search.find_edges_from(graph, nid) + search.find_edges_to(graph, nid):
                impacted_edges[(edge.source, edge.target, edge.type)] = edge
                other = edge.target if edge.source == nid else edge.source
                if other in changed_ids:
                    continue
                affected_ids.add(other)
                if other not in visited:
                    next_frontier.add(other)
                    visited.add(other)
        if not next_frontier:
            break
        max_depth_reached = depth
        frontier = next_frontier

    if affected_ids and max_depth_reached == 0:
        max_depth_reached = 1

    return affected_ids, list(impacted_edges.values()), max_depth_reached


def _match_changed_file_nodes(
    graph: KnowledgeGraph,
    file_path: str,
) -> list[GraphNode]:
    """Match changed files by exact path first, then by normalized suffix."""
    exact = search.find_nodes_by_file(graph, file_path)
    if exact:
        return exact

    normalized = file_path.replace("\\", "/").lower()
    basename = Path(normalized).name
    suffix_matches = [
        node for node in graph.nodes
        if node.file_path and node.file_path.replace("\\", "/").lower().endswith(normalized)
    ]
    if suffix_matches:
        return suffix_matches

    return [
        node for node in graph.nodes
        if node.file_path and Path(node.file_path).name.lower() == basename
    ]


def format_diff_report(ctx: DiffContext) -> str:
    """Format a DiffContext as a human-readable impact report."""
    lines: list[str] = [
        "## Impact Analysis",
        f"Changed files: {', '.join(ctx.changed_files)}",
        f"Changed nodes: {len(ctx.changed_nodes)}",
        f"Affected nodes: {len(ctx.affected_nodes)}",
    ]
    if ctx.affected_layers:
        lines.append("Affected layers: " + ", ".join(l.name for l in ctx.affected_layers))
    if ctx.risk_summary:
        lines.append("\n### Risk signals")
        for r in ctx.risk_summary:
            lines.append(f"- {r}")
    if ctx.unmapped_files:
        lines.append("\n### Unmapped files (not in graph)")
        for f in ctx.unmapped_files:
            lines.append(f"- {f}")
    return "\n".join(lines)
