"""Shared utilities — re-exports from fastcode.utils for new namespace."""
from fastcode.utils import (
    setup_logging,
    load_config,
    resolve_config_paths,
    ensure_dir,
    count_tokens,
    truncate_to_tokens,
    compute_file_hash,
    get_repo_name_from_url,
    clean_docstring,
)

__all__ = [
    "setup_logging",
    "load_config",
    "resolve_config_paths",
    "ensure_dir",
    "count_tokens",
    "truncate_to_tokens",
    "compute_file_hash",
    "get_repo_name_from_url",
    "clean_docstring",
]
