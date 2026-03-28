"""Runtime bootstrap helpers for GraphFirstFacade."""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse
from importlib import import_module

from fastcode.graph.models import ProjectMeta
from fastcode.symbol_backend.hybrid_provider import HybridProvider
from fastcode.utils import get_language_from_extension

from .service_container import ServiceContainer


class GraphFirstRuntimeBootstrapMixin:
    @staticmethod
    def _facade_module():
        return import_module("fastcode.app.graph_first_facade")

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
        facade_module = self._facade_module()
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
        self.repo_indexed = bool(self.project_root and facade_module.graph_exists(self.project_root))
        self.multi_repo_mode = False
        self.loaded_repositories = {}
        self._code_retriever = None
        self.invalidate_index_scan_cache()
        if self.project_root is not None:
            self._container = ServiceContainer(project_root=self.project_root)

    def index_repository(self, force: bool = False) -> None:
        """Build or refresh the persisted graph for the currently loaded repository."""
        facade_module = self._facade_module()
        project_root = self._require_project_root()
        provider = self._make_symbol_provider()
        project_meta = self._build_project_meta(project_root)

        if force or not facade_module.graph_exists(project_root):
            facade_module.build_graph(project_root, provider, project_meta)
        else:
            stale = facade_module.check_staleness(project_root)
            if stale.is_stale and stale.changed_files:
                changed_paths = [project_root / rel for rel in stale.changed_files]
                facade_module.incremental_update(project_root, changed_paths, provider)
            elif stale.is_stale:
                facade_module.build_graph(project_root, provider, project_meta)

        self.repo_indexed = facade_module.graph_exists(project_root)
        self._container = ServiceContainer(project_root=project_root)
        self.invalidate_index_scan_cache()

    def cleanup(self) -> None:
        """Clean up loader-owned temporary resources."""
        self.loader.cleanup()

    def _require_project_root(self) -> Path:
        if self.project_root is None:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        return self.project_root

    def _ensure_container(self) -> ServiceContainer:
        project_root = self._require_project_root()
        if self._container is None or self._container.project_root != project_root:
            self._container = ServiceContainer(project_root=project_root)
        return self._container

    @staticmethod
    def _make_symbol_provider() -> HybridProvider:
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
