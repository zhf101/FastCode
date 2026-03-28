"""Context budget controller — ensures graph context is prioritized over retrieval."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Rough characters-per-token estimate (conservative)
_CHARS_PER_TOKEN = 4


@dataclass
class BudgetAllocation:
    max_tokens: int
    reserved_response_tokens: int
    graph_tokens: int
    retrieval_tokens: int

    @property
    def total_context_tokens(self) -> int:
        return self.max_tokens - self.reserved_response_tokens

    @property
    def available_retrieval_tokens(self) -> int:
        return max(0, self.total_context_tokens - self.graph_tokens)


class ContextBudget:
    """Controls token allocation between graph context and retrieval augmentation.

    Graph context always gets priority. Retrieval is allocated only from
    what remains after graph content is reserved.
    """

    def __init__(
        self,
        max_context_tokens: int = 128_000,
        reserved_response_tokens: int = 4_000,
        graph_budget_fraction: float = 0.6,
    ) -> None:
        if not 0.0 < graph_budget_fraction <= 1.0:
            raise ValueError("graph_budget_fraction must be in (0, 1]")
        self._max = max_context_tokens
        self._reserved = reserved_response_tokens
        self._graph_fraction = graph_budget_fraction

    def allocate(self, graph_text: str, retrieval_text: str) -> BudgetAllocation:
        """Calculate how much of each text fits within budget.

        Graph content is allocated up to *graph_budget_fraction* of the
        available context; retrieval gets the remainder.
        """
        available = self._max - self._reserved
        graph_budget = int(available * self._graph_fraction)
        retrieval_budget = available - graph_budget

        graph_actual = min(graph_budget, self._estimate_tokens(graph_text))
        retrieval_actual = min(retrieval_budget, self._estimate_tokens(retrieval_text))

        alloc = BudgetAllocation(
            max_tokens=self._max,
            reserved_response_tokens=self._reserved,
            graph_tokens=graph_actual,
            retrieval_tokens=retrieval_actual,
        )
        logger.debug(
            "ContextBudget: graph=%d retrieval=%d available=%d",
            graph_actual, retrieval_actual, available,
        )
        return alloc

    def trim_to_budget(self, text: str, token_limit: int) -> tuple[str, bool]:
        """Trim *text* to fit within *token_limit* tokens.

        Returns (trimmed_text, was_truncated).
        """
        char_limit = token_limit * _CHARS_PER_TOKEN
        if len(text) <= char_limit:
            return text, False
        return text[:char_limit].rstrip() + "\n[... truncated ...]", True

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN)
