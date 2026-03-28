"""Graph-first compatibility facade.

Provides a narrow FastCode-like lifecycle around the new graph-first runtime:
`load_repository()`, `index_repository()`, `query()`, and `cleanup()`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastcode.graph.persistence import graph_exists
from fastcode.graph.staleness import check_staleness
from fastcode.graph.models import ProjectMeta
from fastcode.graph_pipeline.incremental_updater import incremental_update
from fastcode.graph_pipeline.runtime import build_graph
from fastcode.retrieval_runtime.code_retriever import CodeRetriever
from fastcode.symbol_backend.hybrid_provider import HybridProvider
from fastcode.utils import get_language_from_extension, load_config, resolve_config_paths, setup_logging

from .query_router import QueryRouter
from .service_container import ServiceContainer
from ..loader import RepositoryLoader


def _default_config(project_root: str) -> dict[str, Any]:
    """Return a small but usable fallback config."""
    return resolve_config_paths(
        {
            "repo_root": "./repos",
            "repository": {
                "clone_depth": 1,
                "max_file_size_mb": 5,
                "backup_directory": "./repo_backup",
                "ignore_patterns": ["*.pyc", "__pycache__", "node_modules", ".git"],
                "supported_extensions": [".py", ".js", ".ts", ".java", ".go"],
            },
            "logging": {
                "level": "INFO",
                "console": True,
                "file": "./logs/fastcode.log",
            },
        },
        project_root,
    )


def _load_runtime_config(config_path: str | None = None) -> dict[str, Any]:
    """Load FastCode config using the same search rules as the legacy runtime."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if config_path:
        if os.path.exists(config_path):
            return load_config(config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    possible_paths = [
        os.path.join(project_root, "config", "config.yaml"),
        os.path.join(project_root, "..", "config", "config.yaml"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return load_config(path)

    return _default_config(project_root)


class GraphFirstFacade:
    """Compatibility layer over the graph-first runtime."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = _load_runtime_config(config_path)
        self.logger = setup_logging(self.config)
        self.loader = RepositoryLoader(self.config)
        self.router = QueryRouter()
        self._code_retriever: CodeRetriever | None = None

        self.repo_loaded = False
        self.repo_indexed = False
        self.repo_info: dict[str, Any] = {}

        self.project_root: Path | None = None
        self._container: ServiceContainer | None = None

    @staticmethod
    def _infer_is_url(source: str) -> bool:
        """Infer whether *source* should be treated as a URL."""
        normalized = (source or "").strip()
        if not normalized:
            return False

        if os.path.exists(normalized):
            return False

        parsed = urlparse(normalized)
        if parsed.scheme in {"http", "https", "ssh", "git", "file"}:
            return True

        return bool(re.match(r"^[^@\s]+@[^:\s]+:[^\s]+$", normalized))

    def load_repository(self, source: str, is_url: bool | None = None, is_zip: bool = False) -> None:
        """Load a repository through the shared RepositoryLoader."""
        resolved_is_url = is_url if is_url is not None else self._infer_is_url(source)

        if is_zip:
            self.loader.load_from_zip(source)
        elif resolved_is_url:
            self.loader.load_from_url(source)
        else:
            self.loader.load_from_path(source)

        self.project_root = Path(self.loader.repo_path).resolve() if self.loader.repo_path else None
        self.repo_info = self.loader.get_repository_info()
        self.repo_loaded = True
        self.repo_indexed = bool(self.project_root and graph_exists(self.project_root))
        self._code_retriever = None
        if self.project_root is not None:
            self._container = ServiceContainer(project_root=self.project_root)

    def index_repository(self, force: bool = False) -> None:
        """Build or refresh the persisted graph for the currently loaded repository."""
        project_root = self._require_project_root()
        provider = self._make_symbol_provider()
        project_meta = self._build_project_meta(project_root)

        if force or not graph_exists(project_root):
            build_graph(project_root, provider, project_meta)
        else:
            stale = check_staleness(project_root)
            if stale.is_stale and stale.changed_files:
                changed_paths = [project_root / rel for rel in stale.changed_files]
                incremental_update(project_root, changed_paths, provider)
            elif stale.is_stale:
                build_graph(project_root, provider, project_meta)

        self.repo_indexed = graph_exists(project_root)
        self._container = ServiceContainer(project_root=project_root)

    def query(self, question: str, **_: Any) -> dict[str, Any]:
        """Run a graph-first query and return a FastCode-compatible payload."""
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        return self._run_query_router(question)

    def cleanup(self) -> None:
        """Clean up loader-owned temporary resources."""
        self.loader.cleanup()

    def _run_query_router(self, question: str) -> dict[str, Any]:
        project_root = self._require_project_root()
        self._ensure_router(project_root)
        container = self._container or ServiceContainer(project_root=project_root)
        result = self.router.route(question, container)
        self._container = container

        repo_name = self.repo_info.get("name") if self.repo_info else None
        return {
            "answer": result.answer.answer,
            "query": question,
            "context_elements": result.answer.context_nodes + result.answer.context_edges,
            "sources": [],
            "searched_repositories": [repo_name] if repo_name else [],
            "graph_ready": result.graph_ready,
            "restricted_mode": result.restricted_mode,
            "intent": result.classification.intent.value,
            "intent_confidence": result.classification.confidence,
        }

    def _ensure_router(self, project_root: Path) -> None:
        if self._code_retriever is not None:
            return
        if not graph_exists(project_root):
            return

        self._code_retriever = CodeRetriever.from_legacy(
            self.config,
            repo_root=project_root,
        )
        self.router = QueryRouter(code_retriever=self._code_retriever)

    def _require_project_root(self) -> Path:
        if self.project_root is None:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        return self.project_root

    def _make_symbol_provider(self) -> HybridProvider:
        return HybridProvider()

    def _build_project_meta(self, project_root: Path) -> ProjectMeta:
        files = self.loader.scan_files()
        languages = sorted(
            {
                get_language_from_extension(Path(file_info["path"]).suffix)
                for file_info in files
                if get_language_from_extension(Path(file_info["path"]).suffix) != "unknown"
            }
        )
        return ProjectMeta(
            name=str(self.repo_info.get("name") or project_root.name),
            languages=languages,
            frameworks=[],
            description=f"Graph-first view of {self.repo_info.get('name', project_root.name)}",
        )
