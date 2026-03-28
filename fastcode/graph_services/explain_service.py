"""Explain service — builds node-focused explanation context from the graph."""
from __future__ import annotations

from dataclasses import dataclass, field

from fastcode.graph.models import GraphEdge, GraphNode, KnowledgeGraph, Layer
from fastcode.graph import search


@dataclass
class ExplainContext:
    target: GraphNode
    layer: Layer | None
    upstream: list[GraphNode]     # nodes that call/import the target
    downstream: list[GraphNode]   # nodes the target calls/imports
    same_file_nodes: list[GraphNode]
    impacted_edges: list[GraphEdge]
    project_name: str


def build_explain_context(
    graph: KnowledgeGraph,
    target_name: str,
) -> ExplainContext | None:
    """Build an ExplainContext for a node matching *target_name*.

    Returns None if no matching node is found.
    """
    matches = search.find_nodes_by_name(graph, target_name)
    if not matches:
        # Partial match fallback
        tl = target_name.lower()
        matches = [n for n in graph.nodes if tl in n.name.lower()]
    if not matches:
        return None

    target = matches[0]

    # Layer containing the target
    layer: Layer | None = None
    for l in graph.layers:
        if target.id in l.node_ids:
            layer = l
            break

    # Upstream: nodes pointing TO target
    upstream_ids = {e.source for e in search.find_edges_to(graph, target.id)}
    upstream = [n for n in graph.nodes if n.id in upstream_ids]

    # Downstream: nodes target points TO
    downstream_ids = {e.target for e in search.find_edges_from(graph, target.id)}
    downstream = [n for n in graph.nodes if n.id in downstream_ids]

    # Same-file siblings
    same_file: list[GraphNode] = []
    if target.file_path:
        same_file = [
            n for n in search.find_nodes_by_file(graph, target.file_path)
            if n.id != target.id
        ]

    impacted_edges = (
        search.find_edges_from(graph, target.id) +
        search.find_edges_to(graph, target.id)
    )

    return ExplainContext(
        target=target,
        layer=layer,
        upstream=upstream,
        downstream=downstream,
        same_file_nodes=same_file,
        impacted_edges=impacted_edges,
        project_name=graph.project.name,
    )


def format_explain_prompt(ctx: ExplainContext) -> str:
    """Format an ExplainContext into a human-readable explanation."""
    lines: list[str] = [
        f"## Explain: {ctx.target.name}",
        f"Project: {ctx.project_name}",
        f"Type: {ctx.target.type}",
    ]
    if ctx.target.file_path:
        lines.append(f"File: {ctx.target.file_path}")
    if ctx.layer:
        lines.append(f"Layer: {ctx.layer.name} — {ctx.layer.description}")
    if ctx.target.summary:
        lines.append(f"Summary: {ctx.target.summary}")

    if ctx.upstream:
        lines.append(f"\nCalled/imported by: {', '.join(n.name for n in ctx.upstream)}")
    if ctx.downstream:
        lines.append(f"Calls/imports: {', '.join(n.name for n in ctx.downstream)}")
    if ctx.same_file_nodes:
        lines.append(
            f"Same file: {', '.join(n.name for n in ctx.same_file_nodes[:5])}"
        )
    return "\n".join(lines)
