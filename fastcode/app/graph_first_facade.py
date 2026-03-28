"""Graph-first compatibility facade."""
from __future__ import annotations

import os
from typing import Any

from fastcode.graph.persistence import graph_exists
from fastcode.graph.staleness import check_staleness
from fastcode.graph_pipeline.incremental_updater import incremental_update
from fastcode.graph_pipeline.runtime import build_graph
from fastcode.utils import load_config, resolve_config_paths, setup_logging

from .facade_adapters import (
    FallbackCacheManager,
    GraphFirstAnswerFormatter,
    GraphFirstGraphBuilderProxy,
    GraphFirstRetrieverProxy,
    GraphFirstVectorStoreProxy,
)
from .facade_query_support import GraphFirstQuerySupportMixin
from .facade_retrieval_support import GraphFirstRetrievalSupportMixin
from .facade_repository_inspection import GraphFirstRepositoryInspectionMixin
from .facade_repository_lifecycle import GraphFirstRepositoryLifecycleMixin
from .facade_repository_state import GraphFirstRepositoryStateMixin
from .facade_runtime_bootstrap import GraphFirstRuntimeBootstrapMixin
from .facade_router_support import GraphFirstRouterSupportMixin
from .facade_session_support import GraphFirstSessionSupportMixin
from ..loader import RepositoryLoader

try:
    from fastcode.cache import CacheManager as _LegacyCacheManager
except ModuleNotFoundError:
    _LegacyCacheManager = None


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


class GraphFirstFacade(
    GraphFirstRuntimeBootstrapMixin,
    GraphFirstRouterSupportMixin,
    GraphFirstQuerySupportMixin,
    GraphFirstRetrievalSupportMixin,
    GraphFirstSessionSupportMixin,
    GraphFirstRepositoryLifecycleMixin,
    GraphFirstRepositoryStateMixin,
    GraphFirstRepositoryInspectionMixin,
):
    """Compatibility layer over the graph-first runtime."""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self.config = _load_runtime_config(config_path)
        self.logger = setup_logging(self.config)
        self.loader = RepositoryLoader(self.config)
        self.router = None
        self._answer_formatter = GraphFirstAnswerFormatter(self)
        self._vector_store_proxy = GraphFirstVectorStoreProxy(self)
        self._retriever_proxy = GraphFirstRetrieverProxy(self)
        self._graph_builder_proxy: GraphFirstGraphBuilderProxy | None = None
        self._cache_manager = (
            _LegacyCacheManager(self.config)
            if _LegacyCacheManager is not None
            else FallbackCacheManager()
        )
        self._index_scan_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._index_scan_cache_ttl = float(
            self.config.get("vector_store", {}).get("index_scan_cache_ttl", 30.0)
        )
        self._code_retriever = None
        self._prompt_builder_answer_generator: Any | None = None

        self.repo_loaded = False
        self.repo_indexed = False
        self.repo_info: dict[str, Any] = {}
        self.multi_repo_mode = False
        self.loaded_repositories: dict[str, Any] = {}

        self.project_root: Path | None = None
        self._container: ServiceContainer | None = None

    @property
    def vector_store(self):
        return self._vector_store_proxy

    @property
    def answer_generator(self):
        return self._answer_formatter

    @property
    def retriever(self):
        return self._retriever_proxy

    @property
    def cache_manager(self):
        return self._cache_manager

    @property
    def graph_builder(self):
        if self._graph_builder_proxy is None:
            self._graph_builder_proxy = GraphFirstGraphBuilderProxy(self)
        else:
            self._graph_builder_proxy._rebuild()
        return self._graph_builder_proxy

    def format_answer_with_sources(self, result: dict[str, Any]) -> str:
        return self._answer_formatter.format_answer_with_sources(result)

    def invalidate_index_scan_cache(self) -> None:
        self._index_scan_cache = None
