"""IterativeRetriever — multi-round retrieval for augmentation scenarios.

Wraps the legacy IterativeAgent behind a simple, safe interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .code_retriever import CodeRetriever, CodeSnippet, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class IterativeRetrievalResult:
    query: str
    rounds: int = 0
    snippets: list[CodeSnippet] = field(default_factory=list)
    error: str | None = None
    available: bool = True
    unavailable_reason: str | None = None
    backend_metadata: dict[str, object] | None = None

    @property
    def found(self) -> bool:
        return bool(self.snippets)


class IterativeRetriever:
    """Performs multi-round retrieval, expanding scope when early rounds are thin.

    Uses a CodeRetriever as the underlying engine. If the retriever is in
    no-op mode, all rounds are no-ops — safe and non-crashing.
    """

    def __init__(
        self,
        retriever: CodeRetriever,
        max_rounds: int = 3,
        min_snippets_per_round: int = 2,
    ) -> None:
        self._retriever = retriever
        self._max_rounds = max(1, max_rounds)
        self._min_snippets = max(1, min_snippets_per_round)

    def retrieve(self, query: str, *, max_results: int = 10) -> IterativeRetrievalResult:
        """Run up to max_rounds of retrieval, stopping early if results are sufficient.

        Always returns an IterativeRetrievalResult — never raises.
        """
        all_snippets: list[CodeSnippet] = []
        seen_paths: set[str] = set()
        rounds = 0

        current_query = query
        backend_metadata: dict[str, object] | None = None
        for _ in range(self._max_rounds):
            rounds += 1
            result: RetrievalResult = self._retriever.retrieve(
                current_query, max_results=max_results,
            )
            backend_metadata = result.backend_metadata
            if result.unavailable:
                logger.warning(
                    "IterativeRetriever: round %d unavailable: %s",
                    rounds,
                    result.unavailable_reason,
                )
                return IterativeRetrievalResult(
                    query=query,
                    rounds=rounds,
                    snippets=all_snippets,
                    available=False,
                    unavailable_reason=result.unavailable_reason,
                    backend_metadata=backend_metadata,
                )
            if result.error:
                logger.warning("IterativeRetriever: round %d error: %s", rounds, result.error)
                break

            new_snippets = [
                s for s in result.snippets
                if s.file_path not in seen_paths
            ]
            for s in new_snippets:
                seen_paths.add(s.file_path)
            all_snippets.extend(new_snippets)

            if len(all_snippets) >= self._min_snippets * rounds:
                break  # sufficient coverage

            # Expand query slightly for next round
            current_query = f"{query} implementation details"

        return IterativeRetrievalResult(
            query=query,
            rounds=rounds,
            snippets=all_snippets,
            available=True,
            backend_metadata=backend_metadata,
        )
