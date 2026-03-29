"""CodeRetriever — minimal retrieval interface wrapping the current retrieval backend.

This module is an augmentation layer. It must not become the primary query entry.
Use only when graph context is insufficient for implementation-level questions.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass
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
    available: bool = True
    unavailable_reason: str | None = None
    backend_metadata: dict[str, Any] | None = None

    @property
    def found(self) -> bool:
        return bool(self.snippets)

    @property
    def unavailable(self) -> bool:
        return not self.available


class CodeRetriever:
    """Thin wrapper that extracts retrieval from the configured HybridRetriever.

    Constructed with an optional backend; if the backend is unavailable,
    all calls return an empty RetrievalResult — no crash, no exception.
    """

    def __init__(
        self,
        backend: Any | None = None,
        *,
        available: bool | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        """
        Args:
            backend: Optional HybridRetriever-compatible object.
                     If None, retrieval is a safe no-op.
        """
        self._backend = backend
        self._available = bool(backend is not None) if available is None else bool(available)
        self._unavailable_reason = unavailable_reason
        if backend is None:
            logger.info("CodeRetriever: no backend provided, running in no-op mode")
            if self._unavailable_reason is None and not self._available:
                self._unavailable_reason = "retrieval backend unavailable"

    @classmethod
    def from_runtime_config(cls, config: dict, repo_root: str | Path | None = None) -> "CodeRetriever":
        """Construct from the current FastCode runtime config.

        Falls back to no-op if retrieval dependencies are unavailable.
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
            return cls(backend, available=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeRetriever.from_runtime_config failed, using no-op: %s", exc)
            return cls(backend=None, available=False, unavailable_reason=str(exc))

    @classmethod
    def from_legacy(cls, config: dict, repo_root: str | Path | None = None) -> "CodeRetriever":
        """Backward-compatible alias for legacy call sites."""
        return cls.from_runtime_config(config, repo_root=repo_root)

    def retrieve(self, query: str, *, max_results: int = 5) -> RetrievalResult:
        """Retrieve code snippets relevant to *query*.

        Always returns a RetrievalResult — never raises.
        """
        if self._backend is None:
            return RetrievalResult(
                query=query,
                available=self._available,
                unavailable_reason=self._unavailable_reason,
                backend_metadata=None,
            )

        try:
            raw = self._call_backend_retrieve(query, max_results=max_results)
            snippets = self._map_raw_results(raw, max_results=max_results)
            return RetrievalResult(
                query=query,
                snippets=snippets,
                available=True,
                backend_metadata=self._backend_metadata(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeRetriever.retrieve failed: %s", exc)
            return RetrievalResult(
                query=query,
                error=str(exc),
                available=True,
                backend_metadata=self._backend_metadata(),
            )

    def _call_backend_retrieve(self, query: str, *, max_results: int):
        """Call backend.retrieve with a signature compatible with known backends."""
        try:
            return self._backend.retrieve(query, top_k=max_results)
        except TypeError:
            return self._backend.retrieve(query)

    def _map_raw_results(self, raw: Any, *, max_results: int) -> list[CodeSnippet]:
        """Normalize backend-specific result payloads into CodeSnippet objects."""
        items = list(raw or [])
        snippets: list[CodeSnippet] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            snippet = self._snippet_from_backend_item(item)
            if snippet is not None:
                snippets.append(snippet)
            if len(snippets) >= max_results:
                break
        return snippets

    def _snippet_from_backend_item(self, item: dict[str, Any]) -> CodeSnippet | None:
        """Map either snippet-style or legacy HybridRetriever-style items."""
        if "element" in item and isinstance(item["element"], dict):
            return self._snippet_from_legacy_hybrid_item(item)
        return self._snippet_from_plain_item(item)

    @staticmethod
    def _snippet_from_plain_item(item: dict[str, Any]) -> CodeSnippet:
        return CodeSnippet(
            file_path=item.get("file_path", ""),
            content=item.get("content", ""),
            score=float(item.get("score", 0.0)),
            line_start=item.get("line_start"),
            line_end=item.get("line_end"),
            symbol_name=item.get("name"),
        )

    @staticmethod
    def _snippet_from_legacy_hybrid_item(item: dict[str, Any]) -> CodeSnippet | None:
        element = item.get("element")
        if not isinstance(element, dict):
            return None

        line_start = element.get("start_line")
        line_end = element.get("end_line")
        if line_start is None and isinstance(element.get("line_range"), (list, tuple)) and len(element["line_range"]) == 2:
            line_start, line_end = element["line_range"]

        content = (
            element.get("code")
            or element.get("content")
            or element.get("summary")
            or ""
        )

        return CodeSnippet(
            file_path=element.get("file_path") or element.get("relative_path", ""),
            content=content,
            score=float(item.get("total_score", item.get("score", 0.0))),
            line_start=line_start,
            line_end=line_end,
            symbol_name=element.get("name"),
            node_id=element.get("id"),
        )

    def _backend_metadata(self) -> dict[str, Any] | None:
        """Extract lightweight metadata from the wrapped backend if available."""
        if self._backend is None:
            return None

        metadata: dict[str, Any] = {}
        reload_result = getattr(self._backend, "last_reload_result", None)
        serialized_reload = self._serialize_backend_value(reload_result)
        if serialized_reload is not None:
            metadata["last_reload_result"] = serialized_reload
        return metadata or None

    @staticmethod
    def _serialize_backend_value(value: Any) -> Any:
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    def is_available(self) -> bool:
        """Return whether a real retrieval backend is available."""
        return self._available and self._backend is not None
