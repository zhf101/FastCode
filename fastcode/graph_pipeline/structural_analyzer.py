"""Structural analyzer — runs symbol extraction across all scanned files."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastcode.symbol_backend.protocol import SymbolProvider
from fastcode.symbol_backend.symbol_index import SymbolIndex

from .scanner import ScanResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    symbol_index: SymbolIndex
    analyzed_files: int = 0
    failed_files: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return self.analyzed_files - len(self.failed_files)


class StructuralAnalyzer:
    """Iterates over scanned files and populates a SymbolIndex via a provider."""

    def analyze(
        self,
        scan_result: ScanResult,
        provider: SymbolProvider,
    ) -> AnalysisResult:
        """Analyze all files in *scan_result* using *provider*.

        Per-file errors are logged and collected; they never abort the run.
        """
        index = SymbolIndex()
        failed: list[str] = []

        for file_path in scan_result.source_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", file_path, exc)
                failed.append(str(file_path))
                continue

            try:
                symbols, relationships = provider.analyze_file(file_path, content)
                index.add(symbols, relationships)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider failed on %s: %s", file_path, exc)
                failed.append(str(file_path))
                continue

        result = AnalysisResult(
            symbol_index=index,
            analyzed_files=len(scan_result.source_files),
            failed_files=failed,
        )
        logger.info(
            "StructuralAnalyzer: %d/%d files ok, %d failed",
            result.success_count,
            result.analyzed_files,
            len(failed),
        )
        return result
