"""Diff service — impact analysis for changed files."""
from __future__ import annotations

from dataclasses import dataclass, field

from fastcode.graph.models import GraphEdge, GraphNode, KnowledgeGraph, Layer
from fastcode.graph import search


@dataclass
class DiffContext:
    changed_files: list[str]
    changed_nodes: list[GraphNode]
    affected_nodes: list[GraphNode]   # one-hop neighbours of changed nodes
    impacted_edges: list[GraphEdge]
    affected_layers: list[Layer]
    unmapped_files: list[str]         # changed files with no matching node
    risk_summary: list[str]


def build_diff_context(
    graph: KnowledgeGraph,
    changed_files: list[str],
) -> DiffContext:
    """Compute impact context for *changed_files*."""
    changed_nodes: list[GraphNode] = []
    unmapped: list[str] = []

    for fp in changed_files:
        matches = search.find_nodes_by_file(graph, fp)
        if matches:
            changed_nodes.extend(matches)
        else:
            unmapped.append(fp)

    changed_ids = {n.id for n in changed_nodes}

    # One-hop affected nodes
    affected_ids: set[str] = set()
    impacted_edges: list[GraphEdge] = []
    for nid in changed_ids:
        for e in search.find_edges_from(graph, nid) + search.find_edges_to(graph, nid):
            impacted_edges.append(e)
            other = e.target if e.source == nid else e.source
            if other not in changed_ids:
                affected_ids.add(other)

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
