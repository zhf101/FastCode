"""Repository state helpers for GraphFirstFacade."""
from __future__ import annotations

from typing import Any


class GraphFirstRepositoryStateMixin:
    def load_multiple_repositories(self, sources: list[dict[str, Any]]) -> None:
        self.logger.info("Loading %d repositories via graph-first flow", len(sources))

        successfully_indexed: list[str] = []
        last_error: Exception | None = None

        for index, source_info in enumerate(sources, start=1):
            source = source_info.get("source")
            is_url = source_info.get("is_url")
            is_zip = source_info.get("is_zip", False)

            try:
                self.logger.info("[%d/%d] Loading repository: %s", index, len(sources), source)
                self.load_repository(source, is_url=is_url, is_zip=is_zip)
                self.index_repository(force=False)
                repo_name = str(self.repo_info.get("name") or "")
                if repo_name:
                    successfully_indexed.append(repo_name)
                    self.logger.info("[%d/%d] Indexed repository: %s", index, len(sources), repo_name)
            except Exception as exc:
                last_error = exc
                self.logger.error("Failed to load repository %s: %s", source, exc)
                continue

        if not successfully_indexed:
            self.multi_repo_mode = False
            self.repo_loaded = False
            self.repo_indexed = False
            self.loaded_repositories = {}
            if last_error is not None:
                raise last_error
            raise RuntimeError("No repositories were successfully indexed.")

        if not self._load_multi_repo_cache(successfully_indexed):
            raise RuntimeError(
                "Repositories were indexed but graph-first multi-repo cache could not be loaded."
            )

    def list_repositories(self) -> list[dict[str, Any]]:
        if self.multi_repo_mode:
            return [
                {
                    "name": repo["name"],
                    "element_count": repo["element_count"],
                    "file_count": repo["file_count"],
                    "size_mb": repo["size_mb"],
                    "url": repo.get("url", "N/A"),
                }
                for repo in self.loaded_repositories.values()
            ]
        if not self.repo_loaded:
            return []
        return [self._current_repository_descriptor()]

    def get_repository_summary(self) -> str:
        if self.multi_repo_mode:
            repo_count = len(self.loaded_repositories)
            total_elements = sum(repo.get("element_count", 0) for repo in self.loaded_repositories.values())
            total_files = sum(repo.get("file_count", 0) for repo in self.loaded_repositories.values())
            summary_parts = [
                f"Repositories: {repo_count}",
                f"Files: {total_files}",
                f"Graph nodes: {total_elements}",
            ]
            if repo_count:
                summary_parts.append("Loaded: " + ", ".join(sorted(self.loaded_repositories)))
            return "\n".join(summary_parts)

        summary_parts = [
            f"Repository: {self.repo_info.get('name', 'Unknown')}",
            f"Files: {self.repo_info.get('file_count', 0)}",
            f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB",
        ]
        if self.repo_indexed and self.project_root is not None:
            summary_parts.append(f"Graph nodes: {len(self._container.ensure_graph().nodes) if self._container else 0}")
        return "\n".join(summary_parts)

    def get_repository_stats(self) -> dict[str, Any]:
        if self.multi_repo_mode:
            repositories = self.list_repositories()
            return {
                "total_repositories": len(repositories),
                "total_elements": sum(repo["element_count"] for repo in repositories),
                "repositories": [
                    {
                        "name": repo["name"],
                        "elements": repo["element_count"],
                        "files": repo["file_count"],
                        "size_mb": repo["size_mb"],
                    }
                    for repo in repositories
                ],
            }

        repositories = self.list_repositories()
        return {
            "total_repositories": len(repositories),
            "total_elements": repositories[0]["element_count"] if repositories else 0,
            "repositories": [
                {
                    "name": repo["name"],
                    "elements": repo["element_count"],
                    "files": repo["file_count"],
                    "size_mb": repo["size_mb"],
                }
                for repo in repositories
            ],
        }

    def _load_multi_repo_cache(self, repo_names: list[str] | None = None) -> bool:
        available = {repo["name"]: repo for repo in self._indexed_repository_entries()}
        names_to_load = repo_names or sorted(available)
        selected = {name: available[name] for name in names_to_load if name in available}

        if not selected:
            return False

        self.multi_repo_mode = True
        self.repo_loaded = True
        self.repo_indexed = True
        self.loaded_repositories = selected
        self.repo_info = {
            "name": "multi-repo",
            "file_count": sum(repo["file_count"] for repo in selected.values()),
            "total_size_mb": sum(repo["size_mb"] for repo in selected.values()),
        }
        self.project_root = None
        self._container = None
        self._code_retriever = None
        return True
