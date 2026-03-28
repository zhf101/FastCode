"""Repository inspection and artifact management for GraphFirstFacade."""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from fastcode.graph.persistence import graph_exists, load_graph
from fastcode.utils import get_language_from_extension


class GraphFirstRepositoryInspectionMixin:
    def get_index_storage_path(self) -> str:
        return str(
            Path(
                self.config.get("vector_store", {}).get("persist_directory", "./data/vector_store")
            ).resolve()
        )

    def is_repository_indexed(self, repo_name: str) -> bool:
        return self.get_indexed_repository(repo_name) is not None

    def get_indexed_repository(self, repo_name: str) -> dict[str, Any] | None:
        for repo in self._indexed_repository_entries():
            if repo.get("name") == repo_name:
                return dict(repo)
        return None

    def get_repository_artifact_status(self, repo_name: str) -> dict[str, Any]:
        persist_dir = Path(self.get_index_storage_path())
        source_root = Path(
            getattr(self.loader, "safe_repo_root", self.config.get("repo_root", "./repos"))
        ).resolve()
        source_path = source_root / repo_name
        index_files: list[dict[str, Any]] = []
        total_index_bytes = 0

        for artifact_name in (
            f"{repo_name}.faiss",
            f"{repo_name}_metadata.pkl",
            f"{repo_name}_bm25.pkl",
            f"{repo_name}_graphs.pkl",
        ):
            artifact_path = persist_dir / artifact_name
            if artifact_path.exists():
                size_bytes = artifact_path.stat().st_size
                total_index_bytes += size_bytes
                index_files.append(
                    {
                        "name": artifact_name,
                        "path": str(artifact_path),
                        "size_bytes": size_bytes,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                    }
                )

        overviews = self._load_repo_overviews_file(persist_dir)
        has_overview = repo_name in overviews
        has_source = source_path.is_dir()

        return {
            "repo_name": repo_name,
            "persist_dir": str(persist_dir),
            "source_root": str(source_root),
            "source_path": str(source_path),
            "index_files": index_files,
            "has_index_files": bool(index_files),
            "has_overview": has_overview,
            "has_source": has_source,
            "total_index_bytes": total_index_bytes,
            "total_index_mb": round(total_index_bytes / (1024 * 1024), 2),
        }

    def find_orphaned_index_files(self) -> dict[str, Any]:
        persist_dir = Path(self.get_index_storage_path())
        if not persist_dir.exists():
            return {
                "persist_dir": str(persist_dir),
                "valid_repositories": [],
                "orphaned_files": [],
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
            }

        files = {path.name: path for path in persist_dir.iterdir() if path.is_file()}
        orphaned_names: set[str] = set()
        valid_repositories: list[str] = []

        for file_name in sorted(name for name in files if name.endswith(".faiss")):
            repo_name = file_name.removesuffix(".faiss")
            metadata_name = f"{repo_name}_metadata.pkl"
            if metadata_name in files:
                valid_repositories.append(repo_name)
            else:
                orphaned_names.add(file_name)

        for file_name in sorted(name for name in files if name.endswith("_metadata.pkl")):
            repo_name = file_name.removesuffix("_metadata.pkl")
            index_name = f"{repo_name}.faiss"
            if index_name not in files:
                orphaned_names.add(file_name)

        orphaned_files = []
        total_size_bytes = 0
        for file_name in sorted(orphaned_names):
            file_path = files[file_name]
            size_bytes = file_path.stat().st_size
            total_size_bytes += size_bytes
            orphaned_files.append(
                {
                    "name": file_name,
                    "path": str(file_path),
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                }
            )

        return {
            "persist_dir": str(persist_dir),
            "valid_repositories": valid_repositories,
            "orphaned_files": orphaned_files,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        }

    def remove_orphaned_index_files(self, file_names: list[str] | None = None) -> dict[str, Any]:
        scan = self.find_orphaned_index_files()
        available = {entry["name"]: entry for entry in scan["orphaned_files"]}
        target_names = file_names or list(available)

        removed_files: list[str] = []
        freed_bytes = 0
        for file_name in target_names:
            entry = available.get(file_name)
            if entry is None:
                continue
            file_path = Path(entry["path"])
            if not file_path.exists():
                continue
            file_path.unlink()
            removed_files.append(file_name)
            freed_bytes += int(entry["size_bytes"])

        if removed_files:
            self.invalidate_index_scan_cache()

        return {
            "persist_dir": scan["persist_dir"],
            "removed_files": removed_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }

    def get_repository_overview(self, repo_name: str) -> dict[str, Any] | None:
        repo_entry = self.get_indexed_repository(repo_name)
        if repo_entry is None:
            return None

        overviews = self._load_repo_overviews_file(Path(self.get_index_storage_path()))
        if repo_name in overviews:
            overview = overviews[repo_name]
            metadata = overview.get("metadata", {}) if isinstance(overview, dict) else {}
            return {
                "repo_name": repo_name,
                "summary": metadata.get("summary") or overview.get("summary", "No summary available."),
                "structure_text": metadata.get("structure_text", ""),
                "file_structure": metadata.get("file_structure", {}),
                "source": "legacy_overview",
            }

        root_path = Path(repo_entry["root_path"])
        graph = load_graph(root_path)
        relative_paths = sorted(
            {
                self._relative_repo_path(root_path, node.file_path)
                for node in graph.nodes
                if node.file_path
            }
        )
        relative_paths = [path for path in relative_paths if path]
        languages = self._count_languages(relative_paths)
        summary = graph.project.description or (
            f"{repo_name} contains {len(relative_paths)} files and {len(graph.nodes)} graph nodes."
        )
        return {
            "repo_name": repo_name,
            "summary": summary,
            "structure_text": self._format_structure_text(relative_paths),
            "file_structure": {
                "languages": languages,
                "total_files": len(relative_paths),
                "all_files": relative_paths,
            },
            "source": "graph",
        }

    def search_indexed_symbols(
        self,
        symbol_name: str,
        *,
        repo_names: list[str] | None = None,
        symbol_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query_lower = symbol_name.lower()
        exact: list[dict[str, Any]] = []
        prefix: list[dict[str, Any]] = []
        contains: list[dict[str, Any]] = []

        for repo in self._selected_index_entries(repo_names):
            root_path = Path(repo["root_path"])
            graph = load_graph(root_path)
            for node in graph.nodes:
                if symbol_type and node.type != symbol_type:
                    continue

                name_lower = node.name.lower()
                payload = self._graph_node_payload(node, repo_name=repo["name"], root_path=root_path)
                if name_lower == query_lower:
                    exact.append(payload)
                elif name_lower.startswith(query_lower):
                    prefix.append(payload)
                elif query_lower in name_lower:
                    contains.append(payload)

        return (exact + prefix + contains)[:limit]

    def get_file_outline(
        self,
        file_path: str,
        *,
        repo_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        candidates: list[tuple[int, str, dict[str, Any], Any]] = []
        normalized_query = file_path.replace("\\", "/")

        for repo in self._selected_index_entries(repo_names):
            root_path = Path(repo["root_path"])
            graph = load_graph(root_path)
            relative_paths = sorted(
                {
                    self._relative_repo_path(root_path, node.file_path)
                    for node in graph.nodes
                    if node.file_path
                }
            )
            for relative_path in relative_paths:
                if not relative_path:
                    continue
                if relative_path == normalized_query:
                    rank = 0
                elif relative_path.endswith(normalized_query):
                    rank = 1
                elif normalized_query in relative_path:
                    rank = 2
                else:
                    continue
                candidates.append((rank, relative_path, repo, graph))

        if not candidates:
            return None

        _, relative_path, repo, graph = sorted(candidates, key=lambda item: (item[0], len(item[1])))[0]
        root_path = Path(repo["root_path"])
        matching_nodes = [
            node
            for node in graph.nodes
            if self._relative_repo_path(root_path, node.file_path) == relative_path
        ]
        file_nodes = [node for node in matching_nodes if node.type == "file"]
        class_nodes = sorted(
            (node for node in matching_nodes if node.type == "class"),
            key=lambda node: node.line_range or (0, 0),
        )
        function_nodes = sorted(
            (node for node in matching_nodes if node.type == "function"),
            key=lambda node: node.line_range or (0, 0),
        )

        class_methods: dict[str, list[str]] = {}
        for class_node in class_nodes:
            methods = [
                node.name
                for node in function_nodes
                if node.metadata.get("parent") == class_node.name
            ]
            class_methods[class_node.id] = methods

        file_stats = self._read_file_stats(root_path / relative_path)
        file_record = file_nodes[0] if file_nodes else None
        return {
            "repo_name": repo["name"],
            "relative_path": relative_path,
            "language": (
                (
                    file_record.metadata.get("language")
                    if file_record is not None and isinstance(file_record.metadata, dict)
                    else None
                )
                or get_language_from_extension(Path(relative_path).suffix)
            ),
            "total_lines": file_stats["total_lines"],
            "code_lines": file_stats["code_lines"],
            "num_imports": file_stats["num_imports"],
            "classes": [
                {
                    "name": node.name,
                    "signature": self._build_node_signature(node, relative_path),
                    "start_line": (node.line_range or (None, None))[0],
                    "end_line": (node.line_range or (None, None))[1],
                    "methods": class_methods.get(node.id, []),
                }
                for node in class_nodes
            ],
            "functions": [
                {
                    "name": node.name,
                    "signature": self._build_node_signature(node, relative_path),
                    "start_line": (node.line_range or (None, None))[0],
                    "end_line": (node.line_range or (None, None))[1],
                    "class_name": node.metadata.get("parent"),
                }
                for node in function_nodes
            ],
        }

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        if use_cache and self._index_scan_cache is not None:
            cache_time, cached = self._index_scan_cache
            if time.time() - cache_time < self._index_scan_cache_ttl:
                return cached

        results = [self._build_index_descriptor(root) for root in self._scan_index_roots()]
        results.sort(key=lambda repo: repo["name"])
        self._index_scan_cache = (time.time(), results)
        return [self._public_repository_descriptor(repo) for repo in results]

    def _indexed_repository_entries(self) -> list[dict[str, Any]]:
        return [self._build_index_descriptor(root) for root in self._scan_index_roots()]

    def _scan_index_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()

        repo_root = Path(
            getattr(self.loader, "safe_repo_root", self.config.get("repo_root", "./repos"))
        ).resolve()
        if repo_root.exists():
            for child in repo_root.iterdir():
                if child.is_dir() and graph_exists(child):
                    key = str(child)
                    if key not in seen:
                        seen.add(key)
                        roots.append(child)

        if self.project_root is not None and graph_exists(self.project_root):
            key = str(self.project_root)
            if key not in seen:
                seen.add(key)
                roots.append(self.project_root)

        return roots

    def _build_index_descriptor(self, project_root: Path) -> dict[str, Any]:
        artifact_root = project_root / ".fastcode"
        artifact_size_mb = round(
            sum(path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()) / (1024 * 1024),
            2,
        ) if artifact_root.exists() else 0.0

        try:
            graph = load_graph(project_root)
            file_paths = {node.file_path for node in graph.nodes if node.file_path}
            current_repo = self._current_repository_descriptor()
            current_repo_name = current_repo["name"] if current_repo else None
            current_repo_url = current_repo["url"] if current_repo else "N/A"

            return {
                "name": graph.project.name or project_root.name,
                "element_count": len(graph.nodes),
                "file_count": len(file_paths),
                "size_mb": artifact_size_mb,
                "url": current_repo_url if graph.project.name == current_repo_name else "N/A",
                "root_path": str(project_root),
            }
        except Exception:
            return {
                "name": project_root.name,
                "element_count": 0,
                "file_count": 0,
                "size_mb": artifact_size_mb,
                "url": "N/A",
                "root_path": str(project_root),
            }

    @staticmethod
    def _public_repository_descriptor(repo: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": repo["name"],
            "element_count": repo["element_count"],
            "file_count": repo["file_count"],
            "size_mb": repo["size_mb"],
            "url": repo.get("url", "N/A"),
        }

    def _current_repository_descriptor(self) -> dict[str, Any]:
        element_count = 0
        if self.repo_indexed and self.project_root is not None and not self.multi_repo_mode:
            container = self._ensure_container()
            element_count = len(container.ensure_graph().nodes)
        return {
            "name": self.repo_info.get("name", self.project_root.name if self.project_root else "Unknown"),
            "element_count": element_count,
            "file_count": self.repo_info.get("file_count", 0),
            "size_mb": self.repo_info.get("total_size_mb", 0),
            "url": self.repo_info.get("url", "N/A"),
        }

    @staticmethod
    def _load_repo_overviews_file(persist_dir: Path) -> dict[str, Any]:
        overview_path = persist_dir / "repo_overviews.pkl"
        if not overview_path.exists():
            return {}

        try:
            with overview_path.open("rb") as handle:
                data = pickle.load(handle)
        except Exception:
            return {}

        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_repo_overviews_file(persist_dir: Path, data: dict[str, Any]) -> None:
        overview_path = persist_dir / "repo_overviews.pkl"
        if data:
            with overview_path.open("wb") as handle:
                pickle.dump(data, handle)
            return
        if overview_path.exists():
            overview_path.unlink()

    def _delete_repo_overview_file(self, persist_dir: Path, repo_name: str) -> bool:
        overviews = self._load_repo_overviews_file(persist_dir)
        if repo_name not in overviews:
            return False
        updated = dict(overviews)
        updated.pop(repo_name, None)
        self._write_repo_overviews_file(persist_dir, updated)
        return True

    def _selected_index_entries(self, repo_names: list[str] | None) -> list[dict[str, Any]]:
        indexed = self._indexed_repository_entries()
        if repo_names is None:
            return indexed
        wanted = set(repo_names)
        return [repo for repo in indexed if repo.get("name") in wanted]

    def _graph_builder_repo_entries(self) -> list[dict[str, Any]]:
        if self.multi_repo_mode:
            return [dict(repo) for repo in self.loaded_repositories.values()]
        if self.project_root is not None and self.repo_indexed:
            current = self.get_indexed_repository(self.repo_info.get("name", ""))
            if current is not None:
                return [current]
            return [
                {
                    "name": self.repo_info.get("name", self.project_root.name),
                    "root_path": str(self.project_root),
                }
            ]
        return []

    def _drop_removed_repository_state(
        self,
        repo_name: str,
        graph_root: Path | None,
        *,
        source_removed: bool,
    ) -> None:
        if self.multi_repo_mode and repo_name in self.loaded_repositories:
            self.loaded_repositories.pop(repo_name, None)
            if not self.loaded_repositories:
                self.multi_repo_mode = False
                self.repo_loaded = False
                self.repo_indexed = False
                self.repo_info = {}
            else:
                self.repo_info = {
                    "name": "multi-repo",
                    "file_count": sum(repo["file_count"] for repo in self.loaded_repositories.values()),
                    "total_size_mb": sum(repo["size_mb"] for repo in self.loaded_repositories.values()),
                }

        if graph_root is not None and self.project_root is not None and graph_root.resolve() == self.project_root.resolve():
            self.repo_indexed = False
            self._container = None
            self._code_retriever = None
            if source_removed:
                self.repo_loaded = False
                self.project_root = None
                self.repo_info = {}

    @staticmethod
    def _directory_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

    @staticmethod
    def _relative_repo_path(root_path: Path, file_path: str | None) -> str:
        if not file_path:
            return ""
        try:
            return Path(file_path).resolve().relative_to(root_path.resolve()).as_posix()
        except Exception:
            return Path(file_path).name

    @staticmethod
    def _count_languages(relative_paths: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in relative_paths:
            language = get_language_from_extension(Path(path).suffix)
            counts[language] = counts.get(language, 0) + 1
        return counts

    @staticmethod
    def _format_structure_text(relative_paths: list[str], limit: int = 40) -> str:
        if not relative_paths:
            return ""
        lines = [f"- {path}" for path in relative_paths[:limit]]
        if len(relative_paths) > limit:
            lines.append(f"- ... ({len(relative_paths) - limit} more)")
        return "\n".join(lines)

    def _graph_node_payload(
        self,
        node: Any,
        *,
        repo_name: str,
        root_path: Path,
    ) -> dict[str, Any]:
        relative_path = self._relative_repo_path(root_path, node.file_path)
        start_line, end_line = (node.line_range or (None, None))
        return {
            "name": node.name,
            "type": node.type,
            "repo_name": repo_name,
            "relative_path": relative_path,
            "start_line": start_line,
            "end_line": end_line,
            "signature": self._build_node_signature(node, relative_path),
            "summary": node.summary,
            "metadata": dict(node.metadata),
        }

    @staticmethod
    def _build_node_signature(node: Any, relative_path: str) -> str | None:
        if node.type == "class":
            return f"class {node.name}"
        if node.type == "function":
            prefix = "async " if node.metadata.get("is_async") else ""
            return f"{prefix}def {node.name}(...)"
        if node.type == "file":
            return relative_path or node.name
        return None

    @staticmethod
    def _read_file_stats(file_path: Path) -> dict[str, int]:
        if not file_path.exists() or not file_path.is_file():
            return {"total_lines": 0, "code_lines": 0, "num_imports": 0}

        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return {"total_lines": 0, "code_lines": 0, "num_imports": 0}

        total_lines = len(lines)
        code_lines = 0
        num_imports = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("import ", "from ", "#", "//", "/*", "*", "--")):
                if stripped.startswith(("import ", "from ")):
                    num_imports += 1
                continue
            code_lines += 1
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "num_imports": num_imports,
        }
