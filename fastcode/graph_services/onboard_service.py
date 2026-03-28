"""Onboard service — structured onboarding guide built from graph, layers, and tour."""
from __future__ import annotations

from dataclasses import dataclass, field

from fastcode.graph.models import GraphNode, KnowledgeGraph, Layer, TourStep
from fastcode.graph import search


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
    # Entry points: file nodes or zero-indegree nodes
    has_incoming = {e.target for e in graph.edges}
    entry_nodes = [
        n for n in graph.nodes
        if n.id not in has_incoming and n.type in ("file", "module", "function")
    ][:5]

    # Key nodes: high complexity or many connections
    degree: dict[str, int] = {}
    for e in graph.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1
    key_nodes = sorted(
        graph.nodes,
        key=lambda n: (n.complexity == "high", degree.get(n.id, 0)),
        reverse=True,
    )[:8]

    return OnboardingContext(
        project_name=graph.project.name,
        description=graph.project.description,
        languages=graph.project.languages,
        frameworks=graph.project.frameworks,
        entry_points=entry_nodes,
        layers=graph.layers,
        tour_steps=graph.tour,
        key_nodes=key_nodes,
    )


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
