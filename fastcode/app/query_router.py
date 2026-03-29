"""Query router — orchestrates intent classification, context building, and answering."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastcode.app.answering import AnswerResult, GraphAnswering
from fastcode.app.intent_classifier import ClassificationResult, IntentClassifier, QueryIntent
from fastcode.app.service_container import ServiceContainer
from fastcode.graph.models import GraphNode, ProjectMeta
from fastcode.graph_services.query_context import QueryContext
from fastcode.retrieval_runtime.code_retriever import CodeRetriever, CodeSnippet
from fastcode.retrieval_runtime.context_budget import ContextBudget
from fastcode.retrieval_runtime.context_packer import ContextPacker
from fastcode.retrieval_runtime.graph_augmented_retriever import GraphAugmentedRetriever
from fastcode.retrieval_runtime.iterative_retriever import IterativeRetriever

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    answer: AnswerResult
    classification: ClassificationResult
    graph_ready: bool
    restricted_mode: bool
    sources: list[dict[str, object]]
    retrieval_available: bool | None = None
    retrieval_unavailable_reason: str | None = None
    retrieval_backend_metadata: dict[str, object] | None = None


_EMPTY_META = ProjectMeta(name="unknown", languages=[], frameworks=[], description="")


class QueryRouter:
    """Orchestrates graph-first query routing.

    Pipeline:
        1. Check graph readiness.
        2. Classify intent.
        3. Dispatch to intent-specific handler.
        4. Produce answer.
    """

    def __init__(
        self,
        *,
        code_retriever: CodeRetriever | None = None,
        packer: ContextPacker | None = None,
    ) -> None:
        retriever = code_retriever or CodeRetriever()
        self._answering = GraphAnswering()
        self._augmenter = GraphAugmentedRetriever(retriever)
        self._iterative = IterativeRetriever(retriever)
        self._packer = packer or ContextPacker(ContextBudget())

    def route(
        self,
        query: str,
        container: ServiceContainer,
        *,
        use_agency_mode: bool = False,
    ) -> RouteResult:
        """Route *query* using services in *container*.

        Always succeeds — errors are surfaced in the answer text.
        """
        ready = container.graph_ready()

        # 1. Classify intent
        classification = container.classifier.classify(query)
        logger.info(
            "QueryRouter: intent=%s confidence=%.2f ready=%s",
            classification.intent, classification.confidence, ready,
        )

        # 2. If graph not ready, return degraded answer
        if not ready:
            ctx = QueryContext(
                query=query, project=_EMPTY_META,
                relevant_nodes=[], relevant_edges=[], relevant_layers=[],
            )
            answer = self._answering.answer(query, ctx, restricted=True)
            return RouteResult(
                answer=answer, classification=classification,
                graph_ready=False, restricted_mode=True,
                sources=[],
                retrieval_available=None,
                retrieval_unavailable_reason=None,
                retrieval_backend_metadata=None,
            )

        # 3. Load graph + dispatch
        graph = container.ensure_graph()
        intent = classification.intent

        if intent == QueryIntent.explain:
            answer, sources = self._handle_explain(query, graph)
        elif intent == QueryIntent.diff:
            answer, sources = self._handle_diff(query, graph)
        elif intent == QueryIntent.onboard:
            answer, sources = self._handle_onboard(query, graph)
            retrieval_available = None
            retrieval_unavailable_reason = None
            retrieval_backend_metadata = None
        else:
            # graph_qa / hybrid_detail / unknown — use general context + augmentation
            ctx = container.context_builder.build(graph, query)
            answer = self._answering.answer(query, ctx, restricted=False)
            answer, retrieval_available, retrieval_unavailable_reason, retrieval_backend_metadata = self._augment_answer(
                query,
                ctx,
                answer,
                use_agency_mode=use_agency_mode,
            )
            sources = self._sources_from_nodes(graph.project.name, ctx.relevant_nodes)
        if intent in {QueryIntent.explain, QueryIntent.diff}:
            retrieval_available = None
            retrieval_unavailable_reason = None
            retrieval_backend_metadata = None

        return RouteResult(
            answer=answer, classification=classification,
            graph_ready=True, restricted_mode=False,
            sources=sources,
            retrieval_available=retrieval_available,
            retrieval_unavailable_reason=retrieval_unavailable_reason,
            retrieval_backend_metadata=retrieval_backend_metadata,
        )

    # ------------------------------------------------------------------
    # Intent-specific handlers
    # ------------------------------------------------------------------

    def _augment_answer(
        self,
        query: str,
        ctx: QueryContext,
        answer: AnswerResult,
        *,
        use_agency_mode: bool,
    ) -> tuple[AnswerResult, bool | None, str | None, dict[str, object] | None]:
        snippets, retrieval_available, retrieval_unavailable_reason, retrieval_backend_metadata = (
            self._agency_augmentation_snippets(query)
            if use_agency_mode
            else self._graph_gap_snippets(query, ctx)
        )
        if not snippets:
            return answer, retrieval_available, retrieval_unavailable_reason, retrieval_backend_metadata

        packed = self._packer.pack(ctx, snippets)
        augmented_text = answer.answer + "\n\n" + packed.text
        return (
            AnswerResult(
                query=answer.query,
                answer=augmented_text,
                context_nodes=answer.context_nodes,
                context_edges=answer.context_edges,
                restricted_mode=answer.restricted_mode,
            ),
            retrieval_available,
            retrieval_unavailable_reason,
            retrieval_backend_metadata,
        )

    def _graph_gap_snippets(
        self,
        query: str,
        ctx: QueryContext,
    ) -> tuple[list[CodeSnippet], bool | None, str | None, dict[str, object] | None]:
        """Return graph-gap snippets when normal augmentation decides they are needed."""
        aug = self._augmenter.augment_from_graph_gap(query, ctx)
        if not aug.triggered or not aug.has_content:
            return [], aug.retrieval_available, aug.retrieval_unavailable_reason, aug.backend_metadata
        return aug.snippets, aug.retrieval_available, aug.retrieval_unavailable_reason, aug.backend_metadata

    def _agency_augmentation_snippets(
        self,
        query: str,
    ) -> tuple[list[CodeSnippet], bool | None, str | None, dict[str, object] | None]:
        """Return iterative retrieval snippets for agency-compatible mode."""
        aug = self._iterative.retrieve(query, max_results=10)
        available = getattr(aug, "available", True)
        unavailable_reason = getattr(aug, "unavailable_reason", None)
        backend_metadata = getattr(aug, "backend_metadata", None)
        if aug.error or not aug.snippets:
            return [], available, unavailable_reason, backend_metadata
        return aug.snippets, available, unavailable_reason, backend_metadata

    def _handle_explain(self, query: str, graph) -> tuple[AnswerResult, list[dict[str, object]]]:
        from fastcode.graph_services.explain_service import (
            build_explain_context, extract_explain_target, format_explain_prompt,
        )
        target = extract_explain_target(query)
        ctx = build_explain_context(graph, target)
        if ctx is None:
            text = f"No node matching {target!r} found in the graph."
            sources: list[dict[str, object]] = []
        else:
            text = format_explain_prompt(ctx)
            sources = self._sources_from_nodes(
                graph.project.name,
                [ctx.target, *ctx.upstream, *ctx.downstream, *ctx.same_file_nodes],
            )
        return AnswerResult(
            query=query, answer=text,
            context_nodes=1 if ctx else 0,
            context_edges=len(ctx.upstream) + len(ctx.downstream) if ctx else 0,
        ), sources

    def _handle_diff(self, query: str, graph) -> tuple[AnswerResult, list[dict[str, object]]]:
        from fastcode.graph_services.diff_service import build_diff_context, format_diff_report
        import re
        # Extract file paths from query (anything ending in known extensions)
        files = re.findall(r"[\w./\\-]+\.(?:py|js|ts|java|go|rb|rs|cpp|c|h)", query)
        ctx = build_diff_context(graph, files)
        text = format_diff_report(ctx)
        sources = self._sources_from_nodes(
            graph.project.name,
            [*ctx.changed_nodes, *ctx.affected_nodes],
        )
        return AnswerResult(
            query=query, answer=text,
            context_nodes=len(ctx.changed_nodes),
            context_edges=len(ctx.impacted_edges),
        ), sources

    def _handle_onboard(self, query: str, graph) -> tuple[AnswerResult, list[dict[str, object]]]:
        from fastcode.graph_services.onboard_service import (
            build_onboarding_context, format_onboarding_guide,
        )
        ctx = build_onboarding_context(graph)
        text = format_onboarding_guide(ctx)
        sources = self._sources_from_nodes(
            graph.project.name,
            [*ctx.entry_points, *ctx.key_nodes],
        )
        return AnswerResult(
            query=query, answer=text,
            context_nodes=len(ctx.key_nodes),
            context_edges=0,
        ), sources

    @staticmethod
    def _sources_from_nodes(
        repository: str,
        nodes: list[GraphNode],
    ) -> list[dict[str, object]]:
        seen: set[tuple[str, str, str, str]] = set()
        sources: list[dict[str, object]] = []
        for index, node in enumerate(nodes):
            file_path = node.file_path or ""
            line_range = ""
            if node.line_range is not None:
                line_range = f"{node.line_range[0]}-{node.line_range[1]}"
            key = (file_path, node.name, node.type, line_range)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "repository": repository,
                    "file": file_path,
                    "name": node.name,
                    "type": node.type,
                    "lines": line_range,
                    "score": max(0.0, 1.0 - (index * 0.05)),
                }
            )
        return sources
