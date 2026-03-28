"""Graph query primitives.

Provides simple, reusable search/traversal helpers over a KnowledgeGraph.
All graph services and the query layer should use these instead of
implementing their own traversal logic.
"""

from __future__ import annotations

from typing import Callable

from .models import EdgeType, GraphEdge, GraphNode, KnowledgeGraph, Layer, NodeType, TourStep


# ---------------------------------------------------------------------------
# Node finders
# ---------------------------------------------------------------------------


def find_nodes_by_name(
    graph: KnowledgeGraph,
    name: str,
    *,
    case_sensitive: bool = False,
) -> list[GraphNode]:
    """Return all nodes whose name matches *name*."""
    if case_sensitive:
        return [n for n in graph.nodes if n.name == name]
    name_lower = name.lower()
    return [n for n in graph.nodes if n.name.lower() == name_lower]


def find_node_by_id(graph: KnowledgeGraph, node_id: str) -> GraphNode | None:
    """Return the node with the given ID, or None."""
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None


def find_nodes_by_type(graph: KnowledgeGraph, node_type: NodeType) -> list[GraphNode]:
    """Return all nodes of a given type."""
    return [n for n in graph.nodes if n.type == node_type]


def find_nodes_by_file(graph: KnowledgeGraph, file_path: str) -> list[GraphNode]:
    """Return all nodes associated with a specific file path."""
    return [n for n in graph.nodes if n.file_path == file_path]


def find_nodes_by_tag(graph: KnowledgeGraph, tag: str) -> list[GraphNode]:
    """Return all nodes that carry a specific tag."""
    return [n for n in graph.nodes if tag in n.tags]


def find_nodes(
    graph: KnowledgeGraph,
    predicate: Callable[[GraphNode], bool],
) -> list[GraphNode]:
    """Return all nodes satisfying an arbitrary predicate."""
    return [n for n in graph.nodes if predicate(n)]


# ---------------------------------------------------------------------------
# Edge finders
# ---------------------------------------------------------------------------


def find_edges_from(graph: KnowledgeGraph, node_id: str) -> list[GraphEdge]:
    """Return all edges originating from *node_id*."""
    return [e for e in graph.edges if e.source == node_id]


def find_edges_to(graph: KnowledgeGraph, node_id: str) -> list[GraphEdge]:
    """Return all edges pointing at *node_id*."""
    return [e for e in graph.edges if e.target == node_id]


def find_edges_between(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> list[GraphEdge]:
    """Return all edges between two specific nodes (either direction)."""
    return [
        e for e in graph.edges
        if (e.source == source_id and e.target == target_id)
        or (e.source == target_id and e.target == source_id)
    ]


def find_edges_by_type(graph: KnowledgeGraph, edge_type: EdgeType) -> list[GraphEdge]:
    """Return all edges of a given type."""
    return [e for e in graph.edges if e.type == edge_type]


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------


def get_neighbours(
    graph: KnowledgeGraph,
    node_id: str,
    *,
    direction: str = "both",
    edge_types: list[EdgeType] | None = None,
) -> list[GraphNode]:
    """Return neighbour nodes of *node_id*.

    Args:
        direction: 'outgoing', 'incoming', or 'both'.
        edge_types: If provided, only consider edges of those types.
    """
    node_ids: set[str] = set()
    for edge in graph.edges:
        if edge_types and edge.type not in edge_types:
            continue
        if direction in ("outgoing", "both") and edge.source == node_id:
            node_ids.add(edge.target)
        if direction in ("incoming", "both") and edge.target == node_id:
            node_ids.add(edge.source)
    return [n for n in graph.nodes if n.id in node_ids]


# ---------------------------------------------------------------------------
# Layer / Tour helpers
# ---------------------------------------------------------------------------


def find_layer_by_id(graph: KnowledgeGraph, layer_id: str) -> Layer | None:
    """Return the layer with the given ID, or None."""
    for layer in graph.layers:
        if layer.id == layer_id:
            return layer
    return None


def get_layer_nodes(graph: KnowledgeGraph, layer_id: str) -> list[GraphNode]:
    """Return all nodes belonging to the specified layer."""
    layer = find_layer_by_id(graph, layer_id)
    if layer is None:
        return []
    id_set = set(layer.node_ids)
    return [n for n in graph.nodes if n.id in id_set]


def get_tour_step(graph: KnowledgeGraph, order: int) -> TourStep | None:
    """Return the TourStep with the given order number, or None."""
    for step in graph.tour:
        if step.order == order:
            return step
    return None


def get_tour_step_nodes(graph: KnowledgeGraph, order: int) -> list[GraphNode]:
    """Return all nodes referenced by a tour step."""
    step = get_tour_step(graph, order)
    if step is None:
        return []
    id_set = set(step.node_ids)
    return [n for n in graph.nodes if n.id in id_set]
