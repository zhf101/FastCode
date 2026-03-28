"""Staleness detection for the FastCode knowledge graph.

A graph is considered stale if:
1. No graph file exists at all.
2. The stored git_commit_hash differs from the current HEAD.
3. Any source file is newer than the graph file on disk.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .persistence import _fastcode_dir, GRAPH_FILE, graph_exists, load_meta


@dataclass
class StalenessResult:
    is_stale: bool
    reason: str
    changed_files: list[str]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _current_git_commit(project_root: Path) -> str | None:
    """Return the current HEAD commit hash, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_changed_files(project_root: Path, since_commit: str) -> list[str]:
    """Return list of files changed since *since_commit* (relative paths)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.splitlines() if f.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


def _files_newer_than_graph(project_root: Path, extensions: tuple[str, ...] = (".py",)) -> list[str]:
    """Return source files modified after the graph file's mtime."""
    graph_file = _fastcode_dir(project_root) / GRAPH_FILE
    if not graph_file.exists():
        return []
    graph_mtime = graph_file.stat().st_mtime
    newer: list[str] = []
    for p in project_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in extensions:
            continue
        # skip .fastcode dir itself
        try:
            p.relative_to(_fastcode_dir(project_root))
            continue
        except ValueError:
            pass
        if p.stat().st_mtime > graph_mtime:
            newer.append(str(p.relative_to(project_root)))
    return newer


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def check_staleness(
    project_root: Path,
    *,
    use_git: bool = True,
    fallback_mtime: bool = True,
) -> StalenessResult:
    """Check whether the persisted graph is stale.

    Strategy:
    1. Missing graph -> stale.
    2. Git available: compare stored commit vs HEAD -> stale if different.
    3. Fallback: compare source file mtimes vs graph file mtime.

    Returns a StalenessResult with is_stale flag, reason string, and
    a list of changed file paths (best-effort).
    """
    if not graph_exists(project_root):
        return StalenessResult(
            is_stale=True,
            reason="no graph file found",
            changed_files=[],
        )

    if use_git:
        current_commit = _current_git_commit(project_root)
        if current_commit is not None:
            try:
                meta = load_meta(project_root)
                stored_commit = meta.git_commit_hash
            except (FileNotFoundError, Exception):
                stored_commit = None

            if stored_commit and current_commit != stored_commit:
                changed = _git_changed_files(project_root, stored_commit)
                return StalenessResult(
                    is_stale=True,
                    reason=f"commit changed: {stored_commit[:8]} -> {current_commit[:8]}",
                    changed_files=changed,
                )

            if stored_commit and current_commit == stored_commit:
                return StalenessResult(
                    is_stale=False,
                    reason="commit unchanged",
                    changed_files=[],
                )

    if fallback_mtime:
        newer = _files_newer_than_graph(project_root)
        if newer:
            return StalenessResult(
                is_stale=True,
                reason=f"{len(newer)} source file(s) newer than graph",
                changed_files=newer,
            )

    return StalenessResult(
        is_stale=False,
        reason="no staleness detected",
        changed_files=[],
    )
