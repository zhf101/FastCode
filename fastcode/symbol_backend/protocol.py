"""Abstract protocol for symbol backend providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SymbolInfo:
    """A discovered symbol (function, class, etc.) in a source file."""
    name: str
    kind: str  # "function", "class", "method", "module"
    file_path: str
    line_start: int
    line_end: int
    parent: str | None = None  # containing class/function name
    is_async: bool = False
    docstring: str | None = None


@dataclass
class RelationshipInfo:
    """A structural relationship between two symbols or files."""
    source: str   # node id or file path
    target: str   # node id or file path
    kind: str     # "imports", "calls", "inherits", "contains"
    file_path: str | None = None


class SymbolProvider(ABC):
    """Abstract base class all symbol backends must implement."""

    @abstractmethod
    def analyze_file(
        self,
        file_path: Path,
        content: str,
    ) -> tuple[list[SymbolInfo], list[RelationshipInfo]]:
        """Analyse a single source file.

        Returns:
            A 2-tuple of (symbols, relationships) discovered in the file.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is ready to serve requests."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
