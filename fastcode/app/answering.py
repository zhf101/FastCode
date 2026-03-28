"""Answering layer — produces graph-based answers without calling LLM."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

from fastcode.graph_services.query_context import QueryContext


@dataclass
class AnswerResult:
    query: str
    answer: str
    context_nodes: int
    context_edges: int
    restricted_mode: bool = False


class GraphAnswering:
    """Synthesises a plain-text answer from a QueryContext.

    This layer is intentionally LLM-free: it formats the graph context
    into a readable structured answer. A later phase can optionally pipe
    this through an LLM for richer prose.
    """

    def answer(self, query: str, ctx: QueryContext, *, restricted: bool = False) -> AnswerResult:
        """Build a structured answer from *ctx* for *query*."""
        if not ctx.relevant_nodes:
            text = (
                "[restricted mode] " if restricted else ""
            ) + f"No graph nodes matched your query: {query!r}. "\
              "Try a full build or refine your search terms."
            return AnswerResult(
                query=query, answer=text,
                context_nodes=0, context_edges=0, restricted_mode=restricted,
            )

        lines: list[str] = []
        if restricted:
            lines.append("[restricted mode — Serena unavailable, showing static graph only]\n")

        lines.append(f"Project: **{ctx.project.name}**")
        lines.append(f"Query: {query!r}\n")

        lines.append(f"## Relevant nodes ({len(ctx.relevant_nodes)})")
        for node in ctx.relevant_nodes:
            loc = f" — {node.file_path}" if node.file_path else ""
            summary = f": {node.summary}" if node.summary else ""
            lines.append(f"- [{node.type}] **{node.name}**{loc}{summary}")

        if ctx.relevant_edges:
            lines.append(f"\n## Relevant relationships ({len(ctx.relevant_edges)})")
            node_name: dict[str, str] = {n.id: n.name for n in ctx.relevant_nodes}
            for edge in ctx.relevant_edges:
                src = node_name.get(edge.source, edge.source)
                tgt = node_name.get(edge.target, edge.target)
                lines.append(f"- {src} --[{edge.type}]--> {tgt}")

        if ctx.relevant_layers:
            lines.append(f"\n## Layers")
            for layer in ctx.relevant_layers:
                lines.append(f"- **{layer.name}**: {layer.description}")

        return AnswerResult(
            query=query,
            answer="\n".join(lines),
            context_nodes=len(ctx.relevant_nodes),
            context_edges=len(ctx.relevant_edges),
            restricted_mode=restricted,
        )
