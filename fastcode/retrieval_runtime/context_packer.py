"""ContextPacker — assembles graph + retrieval context into a single LLM-ready string."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastcode.graph_services.query_context import QueryContext
from .code_retriever import CodeSnippet
from .context_budget import ContextBudget

logger = logging.getLogger(__name__)


@dataclass
class PackedContext:
    text: str
    graph_tokens_used: int
    retrieval_tokens_used: int
    graph_truncated: bool = False
    retrieval_truncated: bool = False
    snippet_count: int = 0


class ContextPacker:
    """Packs graph context and retrieval snippets into a single prompt string.

    Graph section always comes first (highest priority).
    Retrieval section is appended only if budget allows.
    Duplicate snippets (same file + content) are deduplicated.
    """

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self._budget = budget or ContextBudget()

    def pack(
        self,
        graph_context: QueryContext,
        snippets: list[CodeSnippet] | None = None,
    ) -> PackedContext:
        """Produce a packed context string from graph context and optional snippets."""
        snippets = snippets or []

        # --- Graph section ---
        graph_text = graph_context.to_prompt_text()

        # --- Retrieval section ---
        deduped = self._deduplicate(snippets)
        retrieval_text = self._format_snippets(deduped)

        # --- Budget ---
        alloc = self._budget.allocate(graph_text, retrieval_text)

        trimmed_graph, g_trunc = self._budget.trim_to_budget(graph_text, alloc.graph_tokens)
        trimmed_retrieval, r_trunc = self._budget.trim_to_budget(
            retrieval_text, alloc.retrieval_tokens
        )

        parts = [trimmed_graph]
        if trimmed_retrieval.strip():
            parts.append("\n## Retrieval Augmentation\n" + trimmed_retrieval)

        return PackedContext(
            text="\n".join(parts),
            graph_tokens_used=alloc.graph_tokens,
            retrieval_tokens_used=alloc.retrieval_tokens,
            graph_truncated=g_trunc,
            retrieval_truncated=r_trunc,
            snippet_count=len(deduped),
        )

    @staticmethod
    def _deduplicate(snippets: list[CodeSnippet]) -> list[CodeSnippet]:
        seen: set[tuple[str, str]] = set()
        result: list[CodeSnippet] = []
        for s in snippets:
            key = (s.file_path, s.content[:200])
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result

    @staticmethod
    def _format_snippets(snippets: list[CodeSnippet]) -> str:
        if not snippets:
            return ""
        lines: list[str] = []
        for s in snippets:
            loc = s.file_path
            if s.line_start and s.line_end:
                loc += f":{s.line_start}-{s.line_end}"
            lines.append(f"### {loc}\n```\n{s.content}\n```")
        return "\n\n".join(lines)
