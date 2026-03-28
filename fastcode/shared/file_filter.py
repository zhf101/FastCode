"""File filtering utilities for graph pipeline."""
from pathlib import Path
from typing import Sequence

# Default patterns to exclude from analysis
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "*.egg-info",
]

DEFAULT_INCLUDE_EXTENSIONS: list[str] = [
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".rb",
    ".php",
]


def is_excluded(path: Path, exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS) -> bool:
    """Return True if any path component matches an exclude pattern."""
    parts = path.parts
    for pattern in exclude_patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if any(p.startswith(prefix) for p in parts):
                return True
        else:
            if pattern in parts:
                return True
    return False


def is_source_file(
    path: Path,
    include_extensions: Sequence[str] = DEFAULT_INCLUDE_EXTENSIONS,
) -> bool:
    """Return True if the file has a recognised source extension."""
    return path.suffix in include_extensions


def collect_source_files(
    root: Path,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    include_extensions: Sequence[str] = DEFAULT_INCLUDE_EXTENSIONS,
) -> list[Path]:
    """Recursively collect source files under *root*, applying exclude and extension filters."""
    results: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_excluded(p.relative_to(root), exclude_patterns):
            continue
        if is_source_file(p, include_extensions):
            results.append(p)
    return results
