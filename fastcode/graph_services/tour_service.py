"""Tour service — heuristic guided-tour generation from graph."""
from __future__ import annotations

from fastcode.graph.models import GraphEdge, GraphNode, KnowledgeGraph, Layer, TourStep
from fastcode.graph import search


def _zero_indegree_nodes(graph: KnowledgeGraph) -> list[GraphNode]:
    """Return nodes that have no incoming edges (potential entry points)."""
    has_incoming: set[str] = {e.target for e in graph.edges}
    return [n for n in graph.nodes if n.id not in has_incoming]


def generate_heuristic_tour(graph: KnowledgeGraph) -> list[TourStep]:
    """Build a heuristic guided tour.

    Strategy:
    1. If layers exist, emit one step per layer (sorted by name).
    2. Otherwise, start from zero-indegree nodes and walk outgoing edges.
    3. Always produce at least one step for non-empty graphs.
    """
    if not graph.nodes:
        return []

    if graph.layers:
        return _tour_from_layers(graph)
    return _tour_from_topology(graph)


def _tour_from_layers(graph: KnowledgeGraph) -> list[TourStep]:
    steps: list[TourStep] = []
    for order, layer in enumerate(sorted(graph.layers, key=lambda l: l.name), start=1):
        if not layer.node_ids:
            continue
        # Use node names for the description
        names = []
        for nid in layer.node_ids[:5]:  # cap at 5 for readability
            node = search.find_node_by_id(graph, nid)
            if node:
                names.append(node.name)
        desc = f"Explore the {layer.name} layer."
        if names:
            desc += f" Key components: {', '.join(names)}."
        steps.append(TourStep(
            order=order,
            title=layer.name,
            description=desc,
            node_ids=layer.node_ids[:10],
        ))
    return steps if steps else _tour_from_topology(graph)


def _tour_from_topology(graph: KnowledgeGraph) -> list[TourStep]:
    """Fallback: start from zero-indegree nodes, one step per starting node."""
    entry_nodes = _zero_indegree_nodes(graph)
    if not entry_nodes:
        entry_nodes = graph.nodes[:1]  # last resort

    steps: list[TourStep] = []
    seen: set[str] = set()
    order = 1
    for node in entry_nodes[:8]:  # cap number of steps
        if node.id in seen:
            continue
        seen.add(node.id)
        neighbours = search.get_neighbours(graph, node.id, direction="outgoing")
        node_ids = [node.id] + [n.id for n in neighbours[:4] if n.id not in seen]
        steps.append(TourStep(
            order=order,
            title=node.name,
            description=f"Starting point: {node.name}" + (
                f". Connects to: {', '.join(n.name for n in neighbours[:4])}." if neighbours else "."
            ),
            node_ids=node_ids,
        ))
        order += 1
    return steps


def build_tour_prompt(graph: KnowledgeGraph) -> str:
    """Build a prompt string for LLM-based tour enrichment (placeholder)."""
    node_summaries = [
        f"- {n.name} ({n.type}): {n.summary}" for n in graph.nodes[:20]
    ]
    return (
        f"Project: {graph.project.name}\n"
        f"Description: {graph.project.description}\n\n"
        "Nodes:\n" + "\n".join(node_summaries)
    )
