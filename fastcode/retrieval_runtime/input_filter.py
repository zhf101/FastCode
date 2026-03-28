"""InputFilter — filters source files before retrieval to reduce noise.

Shared filter rules with graph_pipeline so both layers stay consistent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastcode.shared.file_filter import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_EXTENSIONS,
    is_excluded,
)

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    accepted: list[Path] = field(default_factory=list)
    rejected: list[Path] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class InputFilter:
    """Filters file paths using the same rules as the graph pipeline scanner.

    This ensures graph pipeline and retrieval runtime stay in sync on what
    constitutes a valid source file.
    """

    def __init__(
        self,
        exclude_patterns: list[str] | None = None,
        include_extensions: list[str] | None = None,
        max_file_size_bytes: int = 1_000_000,  # 1 MB
    ) -> None:
        self._exclude = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self._include = set(include_extensions or DEFAULT_INCLUDE_EXTENSIONS)
        self._max_size = max_file_size_bytes

    def filter(self, paths: list[Path]) -> FilterResult:
        """Partition *paths* into accepted / rejected."""
        result = FilterResult()
        for path in paths:
            if self._accept(path):
                result.accepted.append(path)
            else:
                result.rejected.append(path)
        logger.debug(
            "InputFilter: %d accepted, %d rejected from %d paths",
            result.accepted_count, result.rejected_count, len(paths),
        )
        return result

    def _accept(self, path: Path) -> bool:
        if is_excluded(path, self._exclude):
            return False
        if self._include and path.suffix not in self._include:
            return False
        try:
            if path.is_file() and path.stat().st_size > self._max_size:
                logger.debug("InputFilter: skipping oversized file %s", path)
                return False
        except OSError:
            return False
        return True
