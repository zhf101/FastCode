"""Project scanner — discovers source files for graph pipeline ingestion."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastcode.shared.file_filter import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_EXTENSIONS,
    collect_source_files,
)

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    project_root: Path
    source_files: list[Path] = field(default_factory=list)
    total_files: int = 0
    skipped_files: int = 0

    @property
    def file_paths(self) -> list[str]:
        """Relative file paths as strings (from project_root)."""
        return [str(p.relative_to(self.project_root).as_posix()) for p in self.source_files]


class Scanner:
    """Scans a project directory and returns a ScanResult."""

    def __init__(
        self,
        exclude_patterns: list[str] | None = None,
        include_extensions: list[str] | None = None,
    ) -> None:
        self._exclude = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self._include = include_extensions or DEFAULT_INCLUDE_EXTENSIONS

    def scan(self, project_root: Path) -> ScanResult:
        """Collect all eligible source files under *project_root*."""
        if not project_root.is_dir():
            raise ValueError(f"project_root is not a directory: {project_root}")

        source_files = collect_source_files(
            root=project_root,
            exclude_patterns=self._exclude,
            include_extensions=self._include,
        )

        # Count everything under root to compute skipped
        all_files = [p for p in project_root.rglob("*") if p.is_file()]
        skipped = len(all_files) - len(source_files)

        result = ScanResult(
            project_root=project_root,
            source_files=source_files,
            total_files=len(all_files),
            skipped_files=max(skipped, 0),
        )
        logger.info(
            "Scanner: found %d source files (%d total, %d skipped) in %s",
            len(source_files),
            len(all_files),
            result.skipped_files,
            project_root,
        )
        return result
