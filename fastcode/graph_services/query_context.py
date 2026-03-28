"""Graph query context builder.

Builds a QueryContext from a KnowledgeGraph for a given natural-language query.
Uses graph/search.py primitives only — no inline traversal logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastcode.graph.models import GraphEdge, GraphNode, KnowledgeGraph, Layer, ProjectMeta
from fastcode.graph.search import (
    find_edges_from,
    find_edges_to,
    find_nodes_by_name,
    find_nodes_by_tag,
    get_neighbours,
)


@dataclass
class QueryContext:
    """Assembled context for answering a single query from the graph."""
    query: str
    project: ProjectMeta
    relevant_nodes: list[GraphNode] = field(default_factory=list)
    relevant_edges: list[GraphEdge] = field(default_factory=list)
    relevant_layers: list[Layer] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render context as a compact text block suitable for LLM prompts."""
        lines: list[str] = [
            f"Project: {self.project.name}",
            f"Languages: {', '.join(self.project.languages)}",
            f"Description: {self.project.description}",
            f"Query: {self.query}",
            "",
        ]

        if self.relevant_layers:
            lines.append("Layers:")
            for layer in self.relevant_layers:
                lines.append(f"  [{layer.id}] {layer.name}: {layer.description}")
            lines.append("")

        if self.relevant_nodes:
            lines.append("Relevant nodes:")
            for node in self.relevant_nodes:
                loc = f" ({node.file_path}:{node.line_range[0]}" if node.file_path and node.line_range else ""
                summary = f" — {node.summary}" if node.summary else ""
                lines.append(f"  [{node.type}] {node.name}{loc}{summary}")
            lines.append("")

        if self.relevant_edges:
            lines.append("Relevant relationships:")
            for edge in self.relevant_edges:
                lines.append(f"  {edge.source} --{edge.type}--> {edge.target}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class QueryContextBuilder:
    """Constructs a QueryContext from a KnowledgeGraph for a given query.

    Strategy (v1 — no retrieval augmentation):
    1. Tokenize the query into terms.
    2. Find nodes whose name or tags match any term.
    3. Expand one hop outward.
    4. Collect edges between all retained nodes.
    5. Include any layers that contain a retained node.
    """

    def build(self, graph: KnowledgeGraph, query: str) -> QueryContext:
        terms = _tokenize(query)

        # --- node matching ---
        matched: set[str] = set()
        for term in terms:
            for node in find_nodes_by_name(graph, term):
                matched.add(node.id)
            for node in find_nodes_by_tag(graph, term):
                matched.add(node.id)
            # partial match on name
            t_lower = term.lower()
            for node in graph.nodes:
                if t_lower in node.name.lower() or t_lower in node.summary.lower():
                    matched.add(node.id)

        # --- one-hop expansion ---
        expanded: set[str] = set(matched)
        for node_id in list(matched):
            for neighbour in get_neighbours(graph, node_id):
                expanded.add(neighbour.id)

        node_map = {n.id: n for n in graph.nodes}
        relevant_nodes = [node_map[nid] for nid in expanded if nid in node_map]

        # --- edges between retained nodes ---
        relevant_node_ids = {n.id for n in relevant_nodes}
        relevant_edges = [
            e for e in graph.edges
            if e.source in relevant_node_ids and e.target in relevant_node_ids
        ]

        # --- layers that reference a retained node ---
        relevant_layers = [
            layer for layer in graph.layers
            if any(nid in relevant_node_ids for nid in layer.node_ids)
        ]

        return QueryContext(
            query=query,
            project=graph.project,
            relevant_nodes=relevant_nodes,
            relevant_edges=relevant_edges,
            relevant_layers=relevant_layers,
        )


def _tokenize(query: str) -> list[str]:
    """Split query into searchable terms (length >= 2, no pure punctuation)."""
    import re
    tokens = re.split(r"[\s,/\\.:;()\[\]{}\"']+", query)
    return [t for t in tokens if len(t) >= 2]
