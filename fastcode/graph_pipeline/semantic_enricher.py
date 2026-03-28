"""Semantic enricher — optional LLM-based enrichment of graph node/layer summaries.

Design constraints:
- Only allowed to write to semantic fields: summary, tags, description (layers).
- Must NOT modify structural fields: id, type, name, file_path, line_range, edges.
- LLM failures must degrade gracefully — never abort the pipeline.
"""
from __future__ import annotations

import logging
from typing import Callable

from fastcode.graph.models import GraphNode, KnowledgeGraph, Layer

logger = logging.getLogger(__name__)

# Fields that the enricher is allowed to update on nodes
_ALLOWED_NODE_FIELDS = frozenset({"summary", "tags"})
# Fields allowed on layers
_ALLOWED_LAYER_FIELDS = frozenset({"description"})

# Type alias for a simple LLM callable: (prompt: str) -> str
LLMCallable = Callable[[str], str]


class SemanticEnricher:
    """Applies LLM-generated semantic fields to graph nodes and layers.

    The enricher is deliberately narrow:
    - It receives a KnowledgeGraph and an optional LLM callable.
    - Without an LLM callable it is a no-op (safe fallback).
    - With an LLM callable it asks for summaries for nodes missing them.
    - It validates all LLM output against the whitelist before applying.
    """

    def __init__(self, llm: LLMCallable | None = None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Return an enriched copy of *graph*. Original is never mutated."""
        if self._llm is None:
            logger.debug("SemanticEnricher: no LLM callable, skipping enrichment")
            return graph

        nodes = [self._enrich_node(n) for n in graph.nodes]
        layers = [self._enrich_layer(l) for l in graph.layers]
        return graph.model_copy(update={"nodes": nodes, "layers": layers})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_node(self, node: GraphNode) -> GraphNode:
        # summary == name means the model_validator auto-filled it — treat as missing
        has_real_summary = node.summary and node.summary != node.name
        if has_real_summary:
            return node
        try:
            prompt = (
                f"In one sentence, describe the role of the {node.type} named '{node.name}'"
                + (f" in file '{node.file_path}'" if node.file_path else "") + "."
            )
            raw = self._llm(prompt)  # type: ignore[misc]
            summary = self._sanitize_text(raw)
            if summary:
                return node.model_copy(update={"summary": summary})
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticEnricher: LLM failed for node %s: %s", node.id, exc)
        return node

    def _enrich_layer(self, layer: Layer) -> Layer:
        if layer.description and not layer.description.endswith("(auto-detected)"):
            return layer  # already has a real description
        try:
            prompt = (
                f"In one sentence, describe what the '{layer.name}' layer does in a software project."
            )
            raw = self._llm(prompt)  # type: ignore[misc]
            desc = self._sanitize_text(raw)
            if desc:
                return layer.model_copy(update={"description": desc})
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticEnricher: LLM failed for layer %s: %s", layer.id, exc)
        return layer

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Strip whitespace and limit to a single line."""
        return text.strip().splitlines()[0].strip() if text.strip() else ""
