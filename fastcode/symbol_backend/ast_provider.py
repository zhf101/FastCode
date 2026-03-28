"""AST-based symbol provider using existing FastCode extractors."""
from __future__ import annotations

import logging
from pathlib import Path

from .protocol import RelationshipInfo, SymbolInfo, SymbolProvider

logger = logging.getLogger(__name__)

# Languages supported by tree-sitter parser in this project
_SUPPORTED_EXTENSIONS = {".py"}


class ASTProvider(SymbolProvider):
    """Symbol provider backed by Tree-sitter via FastCode's existing extractors."""

    def __init__(self) -> None:
        self._parser: object | None = None
        self._def_extractor: object | None = None
        self._import_extractor: object | None = None
        self._call_extractor: object | None = None
        self._healthy = False
        self._init_extractors()

    def _init_extractors(self) -> None:
        try:
            from fastcode.tree_sitter_parser import TSParser
            from fastcode.definition_extractor import DefinitionExtractor
            from fastcode.import_extractor import ImportExtractor
            from fastcode.call_extractor import CallExtractor

            self._parser = TSParser()
            self._def_extractor = DefinitionExtractor(self._parser)
            self._import_extractor = ImportExtractor(self._parser)
            self._call_extractor = CallExtractor(self._parser)
            self._healthy = True
        except Exception as exc:
            logger.warning("ASTProvider init failed: %s", exc)
            self._healthy = False

    @property
    def backend_name(self) -> str:
        return "ast"

    def is_available(self) -> bool:
        return self._healthy

    def analyze_file(
        self,
        file_path: Path,
        content: str,
    ) -> tuple[list[SymbolInfo], list[RelationshipInfo]]:
        if not self._healthy:
            return [], []
        if file_path.suffix not in _SUPPORTED_EXTENSIONS:
            return [], []

        symbols: list[SymbolInfo] = []
        relationships: list[RelationshipInfo] = []
        path_str = str(file_path)

        # --- definitions ---
        try:
            defs = self._def_extractor.extract_definitions(content, path_str)  # type: ignore[union-attr]
            for d in defs:
                kind = d.get("type", "function")
                if kind in ("function", "async_function"):
                    sym_kind = "function"
                    is_async = kind == "async_function"
                elif kind == "class":
                    sym_kind = "class"
                    is_async = False
                else:
                    sym_kind = kind
                    is_async = False

                symbols.append(SymbolInfo(
                    name=d["name"],
                    kind=sym_kind,
                    file_path=path_str,
                    line_start=d.get("start_line", 0),
                    line_end=d.get("end_line", 0),
                    parent=d.get("parent"),
                    is_async=is_async,
                    docstring=d.get("docstring"),
                ))
        except Exception as exc:
            logger.warning("Definition extraction failed for %s: %s", file_path, exc)

        # --- imports -> relationships ---
        try:
            imports = self._import_extractor.extract_imports(content, path_str)  # type: ignore[union-attr]
            for imp in imports:
                module = imp.get("module") or imp.get("name", "")
                if module:
                    relationships.append(RelationshipInfo(
                        source=path_str,
                        target=module,
                        kind="imports",
                        file_path=path_str,
                    ))
        except Exception as exc:
            logger.warning("Import extraction failed for %s: %s", file_path, exc)

        # --- calls -> relationships ---
        try:
            calls = self._call_extractor.extract_calls(content, path_str)  # type: ignore[union-attr]
            for call in calls:
                caller = call.get("caller") or path_str
                callee = call.get("name", "")
                if callee:
                    relationships.append(RelationshipInfo(
                        source=caller,
                        target=callee,
                        kind="calls",
                        file_path=path_str,
                    ))
        except Exception as exc:
            logger.warning("Call extraction failed for %s: %s", file_path, exc)

        return symbols, relationships
