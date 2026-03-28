"""Incremental merge of new nodes/edges into an existing KnowledgeGraph.

Merge contract:
- Nodes whose file_path is in changed_files are replaced entirely.
- All edges connected to removed nodes are pruned.
- Nodes from unchanged files are preserved as-is.
- New nodes (from changed files) are appended.
- New edges are appended, duplicates (same source+target+type) are dropped.
- AnalysisMeta fields (git_commit_hash, last_analyzed_at, analysis_mode) must
  be updated by the caller before persisting.
"""

from __future__ import annotations

from .models import GraphEdge, GraphNode, KnowledgeGraph


# ---------------------------------------------------------------------------
# edge dedup key
# ---------------------------------------------------------------------------


def _edge_key(edge: GraphEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.type)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def merge_graph(
    base: KnowledgeGraph,
    new_nodes: list[GraphNode],
    new_edges: list[GraphEdge],
    changed_files: list[str],
) -> KnowledgeGraph:
    """Merge incremental analysis results into *base*.

    Args:
        base: The existing (loaded) KnowledgeGraph.
        new_nodes: Freshly analysed nodes (for changed files only).
        new_edges: Freshly analysed edges (may reference any nodes).
        changed_files: Relative file paths that were re-analysed.

    Returns:
        A new KnowledgeGraph with the merge applied. *base* is not mutated.
    """
    changed_set = set(changed_files)

    # 1. Retain nodes whose file is NOT in changed_files
    retained_nodes: list[GraphNode] = [
        n for n in base.nodes
        if n.file_path is None or n.file_path not in changed_set
    ]
    retained_ids = {n.id for n in retained_nodes}

    # Also track IDs being replaced so we don't double-add
    replaced_ids = {n.id for n in base.nodes if n.file_path in changed_set}

    # 2. Prune edges that touch any removed node
    retained_edges: list[GraphEdge] = [
        e for e in base.edges
        if e.source in retained_ids and e.target in retained_ids
    ]

    # 3. Append new nodes (skip if id already in retained — shouldn't happen,
    #    but guard against accidental duplication)
    merged_node_ids = set(retained_ids)
    final_nodes = list(retained_nodes)
    for node in new_nodes:
        if node.id not in merged_node_ids:
            final_nodes.append(node)
            merged_node_ids.add(node.id)

    # 4. Append new edges, deduplicate by (source, target, type)
    existing_edge_keys = {_edge_key(e) for e in retained_edges}
    final_edges = list(retained_edges)
    for edge in new_edges:
        key = _edge_key(edge)
        if key not in existing_edge_keys:
            # Only add if both endpoints exist in merged graph
            if edge.source in merged_node_ids and edge.target in merged_node_ids:
                final_edges.append(edge)
                existing_edge_keys.add(key)

    return base.model_copy(
        update={
            "nodes": final_nodes,
            "edges": final_edges,
        }
    )
