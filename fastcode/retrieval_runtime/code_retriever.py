"""CodeRetriever — minimal retrieval interface wrapping legacy FastCode HybridRetriever.

This module is an augmentation layer. It must not become the primary query entry.
Use only when graph context is insufficient for implementation-level questions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """A retrieved code snippet with provenance."""
    file_path: str
    content: str
    score: float = 0.0
    line_start: int | None = None
    line_end: int | None = None
    symbol_name: str | None = None
    node_id: str | None = None  # graph node this snippet is linked to


@dataclass
class RetrievalResult:
    query: str
    snippets: list[CodeSnippet] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.snippets)


class CodeRetriever:
    """Thin wrapper that extracts retrieval from the legacy HybridRetriever.

    Constructed with an optional backend; if the backend is unavailable,
    all calls return an empty RetrievalResult — no crash, no exception.
    """

    def __init__(self, backend: Any | None = None) -> None:
        """
        Args:
            backend: Optional HybridRetriever-compatible object.
                     If None, retrieval is a safe no-op.
        """
        self._backend = backend
        if backend is None:
            logger.info("CodeRetriever: no backend provided, running in no-op mode")

    @classmethod
    def from_legacy(cls, config: dict, repo_root: str | Path | None = None) -> "CodeRetriever":
        """Try to construct from legacy FastCode config.

        Falls back to no-op if legacy dependencies are unavailable.
        """
        try:
            from fastcode.vector_store import VectorStore
            from fastcode.embedder import CodeEmbedder
            from fastcode.graph_builder import CodeGraphBuilder

            vs = VectorStore(config)
            emb = CodeEmbedder(config)
            gb = CodeGraphBuilder(config)

            from fastcode.retriever import HybridRetriever
            backend = HybridRetriever(
                config, vs, emb, gb,
                repo_root=str(repo_root) if repo_root else None,
            )
            return cls(backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeRetriever.from_legacy failed, using no-op: %s", exc)
            return cls(backend=None)

    def retrieve(self, query: str, *, max_results: int = 5) -> RetrievalResult:
        """Retrieve code snippets relevant to *query*.

        Always returns a RetrievalResult — never raises.
        """
        if self._backend is None:
            return RetrievalResult(query=query)

        try:
            raw = self._backend.retrieve(query, top_k=max_results)
            snippets = [
                CodeSnippet(
                    file_path=item.get("file_path", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    symbol_name=item.get("name"),
                )
                for item in (raw or [])
            ]
            return RetrievalResult(query=query, snippets=snippets)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeRetriever.retrieve failed: %s", exc)
            return RetrievalResult(query=query, error=str(exc))

    def is_available(self) -> bool:
        """Return True — CodeRetriever always degrades gracefully."""
        return True
