"""Hybrid symbol provider: Serena MCP preferred, AST fallback.

When Serena is available it is the primary source. If Serena is unavailable
(or fails mid-run) the provider transparently falls back to ASTProvider.
Callers receive a unified result regardless of which backend was used.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .ast_provider import ASTProvider
from .protocol import RelationshipInfo, SymbolInfo, SymbolProvider
from .serena_mcp_provider import SerenaMCPProvider

logger = logging.getLogger(__name__)


class HybridProvider(SymbolProvider):
    """Tries Serena first; falls back to AST on failure or unavailability."""

    def __init__(self, mcp_client: Any | None = None) -> None:
        self._serena = SerenaMCPProvider(mcp_client)
        self._ast = ASTProvider()

    @property
    def backend_name(self) -> str:
        if self._serena.is_available():
            return "hybrid"
        return "ast"

    def is_available(self) -> bool:
        return self._ast.is_available()

    @property
    def serena_available(self) -> bool:
        return self._serena.is_available()

    def mark_serena_unavailable(self) -> None:
        """Called by runtime when a mid-run Serena connection drop is detected."""
        self._serena.mark_unavailable()

    def analyze_file(
        self,
        file_path: Path,
        content: str,
    ) -> tuple[list[SymbolInfo], list[RelationshipInfo]]:
        if not self._serena.is_available():
            return self._ast.analyze_file(file_path, content)

        if self._serena.is_available():
            symbols, rels = self._serena.analyze_file(file_path, content)
            # Serena may have marked itself unavailable during the call
            if symbols or rels:
                return symbols, rels
            # Empty result from Serena — fall through to AST
            if not self._serena.is_available():
                logger.warning("Serena became unavailable mid-run; falling back to AST for %s", file_path)

        return self._ast.analyze_file(file_path, content)
