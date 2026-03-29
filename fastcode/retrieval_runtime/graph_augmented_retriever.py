"""GraphAugmentedRetriever — uses graph context to decide when and how to augment.

Pipeline:
    1. Inspect graph context for gaps (abstract nodes, missing summaries).
    2. If augmentation is warranted, call CodeRetriever for implementation details.
    3. Link snippets back to graph nodes where possible.
    4. If augmentation fails, graph answer is returned unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastcode.graph_services.query_context import QueryContext
from .code_retriever import CodeRetriever, CodeSnippet

logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    triggered: bool
    snippets: list[CodeSnippet] = field(default_factory=list)
    reason: str = ""
    error: str | None = None
    retrieval_available: bool | None = None
    retrieval_unavailable_reason: str | None = None

    @property
    def has_content(self) -> bool:
        return bool(self.snippets)


class GraphAugmentedRetriever:
    """Decides whether to augment a graph answer with retrieved code."""

    # Trigger: if more than this fraction of relevant nodes have no real summary,
    # augmentation is likely needed.
    _ABSTRACT_NODE_THRESHOLD = 0.5

    def __init__(self, retriever: CodeRetriever) -> None:
        self._retriever = retriever

    def augment_from_graph_gap(
        self,
        query: str,
        graph_context: QueryContext,
        *,
        max_results: int = 5,
    ) -> AugmentationResult:
        """Attempt retrieval augmentation if the graph context has gaps.

        Args:
            query:          The original user query.
            graph_context:  The graph context already built for this query.
            max_results:    Max snippets to retrieve.

        Returns:
            AugmentationResult — always safe to call regardless of retriever state.
        """
        triggered, reason = self._should_augment(graph_context)
        if not triggered:
            return AugmentationResult(triggered=False, reason=reason, retrieval_available=None)

        logger.info("GraphAugmentedRetriever: augmenting — %s", reason)
        result = self._retriever.retrieve(query, max_results=max_results)

        if result.unavailable:
            return AugmentationResult(
                triggered=True,
                reason=reason,
                error=result.unavailable_reason or "retrieval unavailable",
                retrieval_available=False,
                retrieval_unavailable_reason=result.unavailable_reason or "retrieval unavailable",
            )

        if result.error:
            return AugmentationResult(
                triggered=True,
                reason=reason,
                error=result.error,
                retrieval_available=True,
            )

        # Link snippets to graph nodes by file_path match
        file_to_node: dict[str, str] = {
            n.file_path: n.id
            for n in graph_context.relevant_nodes
            if n.file_path
        }
        for snippet in result.snippets:
            if snippet.node_id is None:
                snippet.node_id = file_to_node.get(snippet.file_path)

        return AugmentationResult(
            triggered=True,
            snippets=result.snippets,
            reason=reason,
            retrieval_available=True,
        )

    def _should_augment(self, ctx: QueryContext) -> tuple[bool, str]:
        """Return (should_augment, reason_str)."""
        if not ctx.relevant_nodes:
            return False, "no relevant nodes — augmentation cannot help"

        # Count nodes whose summary == name (auto-filled, effectively abstract)
        abstract = sum(
            1 for n in ctx.relevant_nodes
            if not n.summary or n.summary == n.name
        )
        ratio = abstract / len(ctx.relevant_nodes)
        if ratio >= self._ABSTRACT_NODE_THRESHOLD:
            return True, (
                f"{abstract}/{len(ctx.relevant_nodes)} nodes lack real summaries"
            )

        return False, "graph context appears sufficient"
