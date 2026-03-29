"""Onboard service — structured onboarding guide built from graph, layers, and tour."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastcode.graph.models import GraphNode, KnowledgeGraph, Layer, TourStep
from fastcode.graph import search
from fastcode.graph_services.tour_service import generate_heuristic_tour


@dataclass
class OnboardingContext:
    project_name: str
    description: str
    languages: list[str]
    frameworks: list[str]
    entry_points: list[GraphNode]
    layers: list[Layer]
    tour_steps: list[TourStep]
    key_nodes: list[GraphNode]       # high-degree or complex nodes worth knowing


def build_onboarding_context(graph: KnowledgeGraph) -> OnboardingContext:
    """Build an onboarding context from the graph, reusing layers and tour."""
    entry_nodes = _collect_entry_points(graph)

    # Key nodes: high complexity or many connections
    degree: dict[str, int] = {}
    for e in graph.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1
    key_nodes = sorted(
        graph.nodes,
        key=lambda n: (n.id in {entry.id for entry in entry_nodes}, n.complexity == "high", degree.get(n.id, 0)),
        reverse=True,
    )[:8]

    tour_steps = graph.tour or generate_heuristic_tour(graph)

    return OnboardingContext(
        project_name=graph.project.name,
        description=graph.project.description,
        languages=graph.project.languages,
        frameworks=graph.project.frameworks,
        entry_points=entry_nodes,
        layers=graph.layers,
        tour_steps=tour_steps,
        key_nodes=key_nodes,
    )


def _collect_entry_points(graph: KnowledgeGraph) -> list[GraphNode]:
    """Collect likely entry points using project metadata first, then topology."""
    matched: list[GraphNode] = []
    seen: set[str] = set()

    for entry in graph.project.entry_points:
        for node in _match_entry_point_nodes(graph, entry):
            if node.id not in seen:
                matched.append(node)
                seen.add(node.id)

    for node in _zero_indegree_entry_nodes(graph):
        if node.id not in seen:
            matched.append(node)
            seen.add(node.id)

    return matched[:5]


def _match_entry_point_nodes(graph: KnowledgeGraph, entry_point: str) -> list[GraphNode]:
    """Match configured project entry points against graph nodes."""
    normalized = entry_point.replace("\\", "/").lower()
    basename = Path(normalized).name
    stem = Path(basename).stem

    exact_path = [
        node for node in graph.nodes
        if node.file_path and node.file_path.replace("\\", "/").lower() == normalized
    ]
    if exact_path:
        return exact_path

    suffix_path = [
        node for node in graph.nodes
        if node.file_path and node.file_path.replace("\\", "/").lower().endswith(normalized)
    ]
    if suffix_path:
        return suffix_path

    basename_matches = [
        node for node in graph.nodes
        if node.file_path and Path(node.file_path).name.lower() == basename
    ]
    if basename_matches:
        return basename_matches

    return [
        node for node in graph.nodes
        if node.name.lower() in {basename, stem}
    ]


def _zero_indegree_entry_nodes(graph: KnowledgeGraph) -> list[GraphNode]:
    """Fallback entry points based on topology and common entry names."""
    has_incoming = {e.target for e in graph.edges}
    candidates = [
        n for n in graph.nodes
        if n.id not in has_incoming and n.type in ("file", "module", "function")
    ]

    def score(node: GraphNode) -> tuple[int, int, str]:
        name = node.name.lower()
        common_entry = int(any(token in name for token in ("main", "app", "server", "index", "cli")))
        type_score = 2 if node.type in ("file", "module") else 1
        return (common_entry, type_score, name)

    return sorted(candidates, key=score, reverse=True)


def format_onboarding_guide(ctx: OnboardingContext) -> str:
    """Format an OnboardingContext as a structured onboarding guide."""
    lines: list[str] = [
        f"# Onboarding: {ctx.project_name}",
        f"{ctx.description}",
    ]
    if ctx.languages:
        lines.append(f"\n**Languages:** {', '.join(ctx.languages)}")
    if ctx.frameworks:
        lines.append(f"**Frameworks:** {', '.join(ctx.frameworks)}")

    if ctx.entry_points:
        lines.append("\n## Entry Points")
        for n in ctx.entry_points:
            lines.append(f"- **{n.name}** ({n.type})" + (f" — {n.file_path}" if n.file_path else ""))

    if ctx.layers:
        lines.append("\n## Architecture Layers")
        for l in ctx.layers:
            lines.append(f"- **{l.name}**: {l.description} ({len(l.node_ids)} nodes)")

    if ctx.tour_steps:
        lines.append("\n## Recommended Reading Order")
        for step in ctx.tour_steps:
            lines.append(f"{step.order}. **{step.title}** — {step.description}")

    if ctx.key_nodes:
        lines.append("\n## Key Components")
        for n in ctx.key_nodes:
            summary = f": {n.summary}" if n.summary and n.summary != n.name else ""
            lines.append(f"- [{n.type}] **{n.name}**{summary}")

    return "\n".join(lines)
