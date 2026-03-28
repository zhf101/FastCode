"""Serena MCP symbol provider.

FastCode does NOT start Serena — it only connects to an already-running
Serena MCP server. If the server is unavailable the provider reports that
clearly so callers can fall back to AST-only mode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .protocol import RelationshipInfo, SymbolInfo, SymbolProvider

logger = logging.getLogger(__name__)


class SerenaMCPProvider(SymbolProvider):
    """Symbol provider that delegates to an external Serena MCP server."""

    def __init__(self, mcp_client: Any | None = None) -> None:
        """
        Args:
            mcp_client: An already-connected MCP client instance, or None.
                        When None the provider starts in unavailable state.
        """
        self._client = mcp_client
        self._available = mcp_client is not None
        if not self._available:
            logger.info("SerenaMCPProvider: no client supplied — operating in unavailable state")

    @property
    def backend_name(self) -> str:
        return "serena_mcp"

    def is_available(self) -> bool:
        return self._available

    def mark_unavailable(self) -> None:
        """Called by runtime when a mid-run connection drop is detected."""
        self._available = False
        logger.warning("SerenaMCPProvider: marked unavailable (connection lost)")

    def analyze_file(
        self,
        file_path: Path,
        content: str,
    ) -> tuple[list[SymbolInfo], list[RelationshipInfo]]:
        if not self._available or self._client is None:
            return [], []

        try:
            raw = self._client.analyze_file(str(file_path), content)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Serena analyze_file failed for %s: %s — marking unavailable", file_path, exc)
            self.mark_unavailable()
            return [], []

        symbols: list[SymbolInfo] = []
        relationships: list[RelationshipInfo] = []

        for s in raw.get("symbols", []):
            try:
                symbols.append(SymbolInfo(
                    name=s["name"],
                    kind=s.get("kind", "function"),
                    file_path=str(file_path),
                    line_start=s.get("line_start", 0),
                    line_end=s.get("line_end", 0),
                    parent=s.get("parent"),
                    is_async=s.get("is_async", False),
                    docstring=s.get("docstring"),
                ))
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed symbol from Serena: %s", exc)

        for r in raw.get("relationships", []):
            try:
                relationships.append(RelationshipInfo(
                    source=r["source"],
                    target=r["target"],
                    kind=r.get("kind", "related"),
                    file_path=str(file_path),
                ))
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed relationship from Serena: %s", exc)

        return symbols, relationships
