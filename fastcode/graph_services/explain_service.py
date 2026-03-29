"""Explain service — builds node-focused explanation context from the graph."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

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


def extract_explain_target(query: str) -> str:
    """Extract the likely explain target from a raw user query."""
    normalized = (query or "").strip()
    normalized = re.sub(
        r"(?i)(explain|解释|说明|describe)[\s:：]*(一下|下)?",
        "",
        normalized,
    ).strip()
    normalized = re.sub(r"^(文件|函数|类|模块)\s+", "", normalized).strip()
    normalized = normalized.strip("`'\" ")
    return normalized or query.strip()


def build_explain_context(
    graph: KnowledgeGraph,
    target_name: str,
) -> ExplainContext | None:
    """Build an ExplainContext for a node matching *target_name*.

    Returns None if no matching node is found.
    """
    normalized_target = extract_explain_target(target_name)
    matches = _match_explain_nodes(graph, normalized_target)
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


def _match_explain_nodes(
    graph: KnowledgeGraph,
    target: str,
) -> list[GraphNode]:
    """Match explain targets by file path, file name, stem, then symbol name."""
    if not target:
        return []

    normalized_target = target.replace("\\", "/").lower()
    target_basename = Path(normalized_target).name
    target_stem = Path(target_basename).stem

    exact_path_matches = [
        node for node in graph.nodes
        if node.file_path and node.file_path.replace("\\", "/").lower() == normalized_target
    ]
    if exact_path_matches:
        return exact_path_matches

    suffix_path_matches = [
        node for node in graph.nodes
        if node.file_path and node.file_path.replace("\\", "/").lower().endswith(normalized_target)
    ]
    if suffix_path_matches:
        return suffix_path_matches

    basename_matches = [
        node for node in graph.nodes
        if node.file_path and Path(node.file_path).name.lower() == target_basename
    ]
    if basename_matches:
        return basename_matches

    stem_matches = [
        node for node in graph.nodes
        if (
            (node.file_path and Path(node.file_path).stem.lower() == target_stem)
            or node.name.lower() == target_stem
        )
    ]
    if stem_matches:
        return stem_matches

    exact_name_matches = search.find_nodes_by_name(graph, target)
    if exact_name_matches:
        return exact_name_matches

    partial = normalized_target.lower()
    return [
        node for node in graph.nodes
        if partial in node.name.lower()
        or (node.file_path and partial in node.file_path.replace("\\", "/").lower())
    ]


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
