"""Repository lifecycle helpers for GraphFirstFacade."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from importlib import import_module


class GraphFirstRepositoryLifecycleMixin:
    @staticmethod
    def _facade_module():
        return import_module("fastcode.app.graph_first_facade")

    def incremental_reindex(self, repo_name: str, repo_path: str | None = None) -> dict[str, Any]:
        facade_module = self._facade_module()
        indexed_repo = self.get_indexed_repository(repo_name)
        resolved_repo_path = Path(repo_path).resolve() if repo_path else None

        if resolved_repo_path is None and indexed_repo is not None:
            resolved_repo_path = Path(indexed_repo["root_path"]).resolve()

        if resolved_repo_path is None or not resolved_repo_path.is_dir():
            return {"status": "path_not_found", "changes": 0}

        if indexed_repo is None and not facade_module.graph_exists(resolved_repo_path):
            return {"status": "no_manifest", "changes": 0}

        stale = facade_module.check_staleness(resolved_repo_path)
        if not stale.is_stale:
            return {"status": "no_changes", "changes": 0}

        self.load_repository(str(resolved_repo_path), is_url=False)
        self.index_repository(force=False)

        changed_files = stale.changed_files
        change_count = len(changed_files) if changed_files else 1
        current_repo = self._current_repository_descriptor()
        file_count = int(current_repo.get("file_count", 0))
        return {
            "status": "success",
            "changes": change_count,
            "added_files": 0,
            "modified_files": change_count,
            "deleted_files": 0,
            "unchanged_files": max(file_count - len(changed_files), 0),
            "total_elements": int(current_repo.get("element_count", 0)),
            "new_elements": 0,
            "preserved_elements": max(int(current_repo.get("element_count", 0)) - change_count, 0),
        }

    def remove_repository(self, repo_name: str, delete_source: bool = True) -> dict[str, Any]:
        artifact_status = self.get_repository_artifact_status(repo_name)
        repo_entry = self.get_indexed_repository(repo_name)
        deleted_files: list[str] = []
        freed_bytes = 0

        for artifact in artifact_status["index_files"]:
            artifact_path = Path(artifact["path"])
            if not artifact_path.exists():
                continue
            artifact_path.unlink()
            deleted_files.append(artifact["name"])
            freed_bytes += int(artifact["size_bytes"])

        persist_dir = Path(self.get_index_storage_path())
        if self._delete_repo_overview_file(persist_dir, repo_name):
            deleted_files.append("repo_overviews.pkl (entry)")

        graph_root = Path(repo_entry["root_path"]) if repo_entry is not None else None
        graph_dir = graph_root / ".fastcode" if graph_root is not None else None
        source_path = Path(artifact_status["source_path"])
        source_removed = False

        if delete_source and source_path.is_dir():
            dir_size = self._directory_size(source_path)
            shutil.rmtree(source_path)
            deleted_files.append(f"repos/{repo_name}/")
            freed_bytes += dir_size
            source_removed = True
        elif graph_dir is not None and graph_dir.exists():
            graph_size = self._directory_size(graph_dir)
            shutil.rmtree(graph_dir)
            deleted_files.append(f"{repo_name}/.fastcode/")
            freed_bytes += graph_size

        self.invalidate_index_scan_cache()
        self._drop_removed_repository_state(repo_name, graph_root, source_removed=source_removed)

        return {
            "repo_name": repo_name,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }
