"""In-memory symbol index built from provider results.

The SymbolIndex aggregates all SymbolInfo and RelationshipInfo produced
during a scan so that the structural_analyzer and assembler can look up
symbols without re-running file analysis.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .protocol import RelationshipInfo, SymbolInfo


class SymbolIndex:
    """Aggregated symbol and relationship store for a single analysis run."""

    def __init__(self) -> None:
        self._symbols: list[SymbolInfo] = []
        self._relationships: list[RelationshipInfo] = []
        # Lookup caches
        self._by_file: dict[str, list[SymbolInfo]] = defaultdict(list)
        self._by_name: dict[str, list[SymbolInfo]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, symbols: list[SymbolInfo], relationships: list[RelationshipInfo]) -> None:
        """Add results from a single file analysis."""
        for s in symbols:
            self._symbols.append(s)
            self._by_file[s.file_path].append(s)
            self._by_name[s.name].append(s)
        self._relationships.extend(relationships)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def symbols_for_file(self, file_path: str | Path) -> list[SymbolInfo]:
        return list(self._by_file[str(file_path)])

    def symbols_by_name(self, name: str) -> list[SymbolInfo]:
        return list(self._by_name[name])

    def relationships_from(self, source: str) -> list[RelationshipInfo]:
        return [r for r in self._relationships if r.source == source]

    def relationships_of_kind(self, kind: str) -> list[RelationshipInfo]:
        return [r for r in self._relationships if r.kind == kind]

    @property
    def all_symbols(self) -> list[SymbolInfo]:
        return list(self._symbols)

    @property
    def all_relationships(self) -> list[RelationshipInfo]:
        return list(self._relationships)

    def __len__(self) -> int:
        return len(self._symbols)
