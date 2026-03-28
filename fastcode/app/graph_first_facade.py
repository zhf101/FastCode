"""Graph-first compatibility facade.

Provides a narrow FastCode-like lifecycle around the new graph-first runtime:
`load_repository()`, `index_repository()`, `query()`, and `cleanup()`.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from fastcode.graph.persistence import graph_exists, load_graph
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


class _GraphFirstAnswerFormatter:
    """Lightweight formatter compatible with the legacy display contract."""

    def __init__(self, facade: "GraphFirstFacade") -> None:
        self._facade = facade

    def format_answer_with_sources(self, result: dict[str, Any]) -> str:
        output = ["## Answer\n", result.get("answer", "")]

        sources = result.get("sources", [])
        if sources:
            output.append("\n\n## Sources\n")
            for index, source in enumerate(sources, 1):
                repository = source.get("repository") or source.get("repo") or ""
                repository_prefix = f"[{repository}] " if repository else ""

                name = source.get("name") or source.get("file") or "unknown"
                source_type = source.get("type") or "node"
                file_path = source.get("file") or source.get("relative_path") or ""
                lines = source.get("lines") or ""
                location = f" in `{file_path}`" if file_path else ""
                if lines:
                    location += f" (lines {lines})"

                score = source.get("score")
                score_text = ""
                if isinstance(score, (int, float)):
                    score_text = f" - Relevance: {float(score):.2f}"

                output.append(
                    f"{index}. {repository_prefix}**{name}** ({source_type}){location}{score_text}"
                )

        if "prompt_tokens" in result:
            output.append(
                f"\n\n*Used {result['prompt_tokens']} prompt tokens, "
                f"{result.get('context_elements', 0)} code snippets*"
            )

        return "\n".join(output)

    def __getattr__(self, name: str):
        return getattr(self._facade._ensure_legacy_runtime().answer_generator, name)


class _GraphFirstVectorStoreProxy:
    """Expose lightweight graph-first index scanning without eager legacy init."""

    def __init__(self, facade: "GraphFirstFacade") -> None:
        self._facade = facade

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        if self._facade.multi_repo_mode:
            return self._facade._ensure_legacy_runtime().vector_store.scan_available_indexes(
                use_cache=use_cache
            )
        return self._facade.scan_available_indexes(use_cache=use_cache)

    def invalidate_scan_cache(self) -> None:
        self._facade.invalidate_index_scan_cache()

    def get_repository_names(self) -> list[str]:
        if self._facade.multi_repo_mode:
            return self._facade._ensure_legacy_runtime().vector_store.get_repository_names()
        return [repo["name"] for repo in self._facade.scan_available_indexes(use_cache=True)]

    def get_count(self) -> int:
        if self._facade.multi_repo_mode:
            return self._facade._ensure_legacy_runtime().vector_store.get_count()
        if not self._facade.repo_indexed:
            return 0
        container = self._facade._ensure_container()
        return len(container.ensure_graph().nodes)

    def __getattr__(self, name: str):
        return getattr(self._facade._ensure_legacy_runtime().vector_store, name)


class _FallbackCacheManager:
    """Small in-memory cache used when optional disk cache deps are unavailable."""

    def __init__(self) -> None:
        self.enabled = True
        self._turns: dict[str, dict[int, dict[str, Any]]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}

    def clear(self) -> bool:
        self._turns.clear()
        self._sessions.clear()
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "memory",
            "size": 0,
            "items": sum(len(turns) for turns in self._turns.values()),
        }

    def save_dialogue_turn(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        answer: str,
        summary: str,
        retrieved_elements: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        turns = self._turns.setdefault(session_id, {})
        turns[turn_number] = {
            "session_id": session_id,
            "turn_number": turn_number,
            "timestamp": time.time(),
            "query": query,
            "answer": answer,
            "summary": summary,
            "retrieved_elements": retrieved_elements or [],
            "metadata": metadata or {},
        }

        session = self._sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "created_at": time.time(),
                "total_turns": 0,
                "last_updated": time.time(),
                "multi_turn": False,
            },
        )
        session["total_turns"] = max(session["total_turns"], turn_number)
        session["last_updated"] = time.time()
        if (metadata or {}).get("multi_turn") is True:
            session["multi_turn"] = True
        return True

    def get_dialogue_turn(self, session_id: str, turn_number: int) -> dict[str, Any] | None:
        return self._turns.get(session_id, {}).get(turn_number)

    def get_dialogue_history(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, Any]]:
        turns = self._turns.get(session_id, {})
        ordered = [turns[number] for number in sorted(turns)]
        if max_turns is None or max_turns >= len(ordered):
            return ordered
        return ordered[-max_turns:]

    def _get_session_index(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._turns.pop(session_id, None)
        return existed

    def list_sessions(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(session) for session in self._sessions.values()),
            key=lambda session: (session.get("created_at", 0), session.get("last_updated", 0)),
            reverse=True,
        )


class GraphFirstFacade:
    """Compatibility layer over the graph-first runtime."""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self.config = _load_runtime_config(config_path)
        self.logger = setup_logging(self.config)
        self.loader = RepositoryLoader(self.config)
        self.router = QueryRouter()
        self._answer_formatter = _GraphFirstAnswerFormatter(self)
        self._vector_store_proxy = _GraphFirstVectorStoreProxy(self)
        self._cache_manager = (
            _LegacyCacheManager(self.config)
            if _LegacyCacheManager is not None
            else _FallbackCacheManager()
        )
        self._index_scan_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._index_scan_cache_ttl = float(
            self.config.get("vector_store", {}).get("index_scan_cache_ttl", 30.0)
        )
        self._code_retriever: CodeRetriever | None = None
        self._legacy_runtime: Any | None = None
        self._legacy_load_synced = False
        self._legacy_index_synced = False

        self.repo_loaded = False
        self.repo_indexed = False
        self.repo_info: dict[str, Any] = {}
        self.multi_repo_mode = False
        self.loaded_repositories: dict[str, Any] = {}

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
        self.multi_repo_mode = False
        self.loaded_repositories = {}
        self._code_retriever = None
        self._legacy_load_synced = False
        self._legacy_index_synced = False
        self.invalidate_index_scan_cache()
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
        self._legacy_index_synced = False
        self.invalidate_index_scan_cache()

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[..., str] | None = None,
    ) -> dict[str, Any]:
        """Run a graph-first query and return a FastCode-compatible payload."""
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        resolved_multi_turn = bool(enable_multi_turn)
        if self._should_use_legacy_query(
            repo_filter=repo_filter,
            session_id=session_id,
            enable_multi_turn=resolved_multi_turn,
            use_agency_mode=use_agency_mode,
            prompt_builder=prompt_builder,
        ):
            return self._query_via_legacy(
                question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=resolved_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )
        result = self._run_query_router(question)
        self._record_dialogue_turn(
            question=question,
            result=result,
            session_id=session_id,
            enable_multi_turn=resolved_multi_turn,
            repo_filter=repo_filter,
        )
        return result

    def query_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        session_id: str | None = None,
        enable_multi_turn: bool | None = None,
        use_agency_mode: bool | None = None,
        prompt_builder: Callable[..., str] | None = None,
    ) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
        """Stream graph-first answers for simple single-repo queries."""
        resolved_multi_turn = bool(enable_multi_turn)
        if self._should_use_legacy_query(
            repo_filter=repo_filter,
            session_id=session_id,
            enable_multi_turn=resolved_multi_turn,
            use_agency_mode=use_agency_mode,
            prompt_builder=prompt_builder,
        ):
            yield from self._query_stream_via_legacy(
                question,
                filters=filters,
                repo_filter=repo_filter,
                session_id=session_id,
                enable_multi_turn=resolved_multi_turn,
                use_agency_mode=use_agency_mode,
                prompt_builder=prompt_builder,
            )
            return

        yield from self._query_stream_graph_first(
            question,
            session_id=session_id,
            enable_multi_turn=resolved_multi_turn,
            repo_filter=repo_filter,
        )

    def load_multiple_repositories(self, sources: list[dict[str, Any]]) -> None:
        runtime = self._ensure_legacy_runtime()
        runtime.load_multiple_repositories(sources)
        self._sync_state_from_legacy()

    def list_repositories(self) -> list[dict[str, Any]]:
        if self.multi_repo_mode:
            runtime = self._sync_legacy_index() if self.repo_indexed else self._ensure_legacy_runtime()
            return runtime.list_repositories()
        if not self.repo_loaded:
            return []
        return [self._current_repository_descriptor()]

    def get_repository_summary(self) -> str:
        if self.multi_repo_mode:
            return self._ensure_legacy_runtime().get_repository_summary()

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
            runtime = self._sync_legacy_index() if self.repo_indexed else self._ensure_legacy_runtime()
            return runtime.get_repository_stats()

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
        runtime = self._ensure_legacy_runtime()
        success = runtime._load_multi_repo_cache(repo_names=repo_names)
        self._sync_state_from_legacy()
        return success

    def incremental_reindex(self, repo_name: str, repo_path: str | None = None) -> dict[str, Any]:
        runtime = self._ensure_legacy_runtime()
        result = runtime.incremental_reindex(repo_name, repo_path=repo_path)
        self._sync_state_from_legacy()
        return result

    def remove_repository(self, repo_name: str, delete_source: bool = True) -> dict[str, Any]:
        runtime = self._ensure_legacy_runtime()
        result = runtime.remove_repository(repo_name, delete_source=delete_source)
        self._sync_state_from_legacy()
        return result

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = self._cache_manager.list_sessions()
        enriched_sessions: list[dict[str, Any]] = []
        for session in sessions:
            enriched = dict(session)
            session_id = enriched.get("session_id", "")
            first_turn = self._cache_manager.get_dialogue_turn(session_id, 1) if session_id else None
            if first_turn:
                query = first_turn.get("query", "")
                enriched["title"] = query[:77] + "..." if len(query) > 80 else query
            else:
                enriched["title"] = f"Session {session_id}" if session_id else "Unknown Session"
            enriched_sessions.append(enriched)
        return enriched_sessions

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return self._cache_manager.get_dialogue_history(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._cache_manager.delete_session(session_id)

    def _get_next_turn_number(self, session_id: str) -> int:
        session_index = self._cache_manager._get_session_index(session_id)
        if session_index:
            return session_index.get("total_turns", 0) + 1
        return 1

    def cleanup(self) -> None:
        """Clean up loader-owned temporary resources."""
        self.loader.cleanup()
        if self._legacy_runtime is not None:
            self._legacy_runtime.cleanup()

    @property
    def vector_store(self):
        return self._vector_store_proxy

    @property
    def answer_generator(self):
        return self._answer_formatter

    @property
    def retriever(self):
        runtime = self._sync_legacy_index() if self.repo_indexed else self._ensure_legacy_runtime()
        return runtime.retriever

    @property
    def cache_manager(self):
        return self._cache_manager

    def format_answer_with_sources(self, result: dict[str, Any]) -> str:
        return self._answer_formatter.format_answer_with_sources(result)

    def _run_query_router(self, question: str) -> dict[str, Any]:
        project_root = self._require_project_root()
        self._ensure_router(project_root)
        container = self._ensure_container()
        result = self.router.route(question, container)
        self._container = container

        repo_name = self.repo_info.get("name") if self.repo_info else None
        return {
            "answer": result.answer.answer,
            "query": question,
            "context_elements": result.answer.context_nodes + result.answer.context_edges,
            "sources": result.sources,
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

    def _should_use_legacy_query(
        self,
        *,
        repo_filter: list[str] | None,
        session_id: str | None,
        enable_multi_turn: bool | None,
        use_agency_mode: bool | None,
        prompt_builder: Callable[..., str] | None,
    ) -> bool:
        if self.multi_repo_mode:
            return True
        if repo_filter is not None:
            current_repo = self.repo_info.get("name")
            if repo_filter != [current_repo]:
                return True
        if enable_multi_turn:
            return True
        if use_agency_mode is not None:
            return True
        if prompt_builder is not None:
            return True
        return False

    def _ensure_legacy_runtime(self):
        if self._legacy_runtime is None:
            from fastcode.main import FastCode as LegacyFastCode

            self._legacy_runtime = LegacyFastCode(config_path=self._config_path)
        return self._legacy_runtime

    def _sync_legacy_load(self):
        runtime = self._ensure_legacy_runtime()
        if self.multi_repo_mode:
            return runtime
        if self._legacy_load_synced or not self.repo_loaded or self.project_root is None:
            return runtime

        runtime.loader.repo_path = str(self.project_root)
        runtime.loader.repo_name = self.loader.repo_name
        runtime.repo_loaded = self.repo_loaded
        runtime.repo_info = dict(self.repo_info)
        runtime.multi_repo_mode = False
        runtime.loaded_repositories = {}
        runtime.config["repo_root"] = str(self.project_root)
        runtime.retriever.set_repo_root(str(self.project_root))
        self._legacy_load_synced = True
        return runtime

    def _sync_legacy_index(self, force: bool = False):
        runtime = self._sync_legacy_load()
        if self.multi_repo_mode:
            return runtime
        if not self.repo_indexed:
            return runtime
        if not self._legacy_index_synced or force:
            runtime.index_repository(force=force)
            self._legacy_index_synced = True
            self._sync_state_from_legacy()
        return runtime

    def _query_via_legacy(self, question: str, **kwargs: Any) -> dict[str, Any]:
        runtime = self._sync_legacy_index()
        return runtime.query(question, **kwargs)

    def _query_stream_via_legacy(self, question: str, **kwargs: Any):
        runtime = self._sync_legacy_index()
        yield from runtime.query_stream(question, **kwargs)

    def _query_stream_graph_first(
        self,
        question: str,
        *,
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
    ) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
        if not self.repo_loaded:
            yield None, {"error": "No repository loaded. Call load_repository() first."}
            return
        if not self.repo_indexed:
            yield None, {"error": "Repository not indexed. Call index_repository() first."}
            return

        try:
            yield None, {"status": "retrieving", "query": question}

            result = self._run_query_router(question)

            yield None, {
                "status": "generating",
                "sources": result.get("sources", []),
                "context_elements": result.get("context_elements", 0),
                "query": question,
            }

            answer = result.get("answer", "")
            for chunk in self._stream_answer_chunks(answer):
                yield chunk, None

            self._record_dialogue_turn(
                question=question,
                result=result,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                repo_filter=repo_filter,
            )

            yield None, {
                "status": "complete",
                "answer": answer,
                "query": result.get("query", question),
                "sources": result.get("sources", []),
                "context_elements": result.get("context_elements", 0),
                "searched_repositories": result.get("searched_repositories", []),
                "intent": result.get("intent"),
                "restricted_mode": result.get("restricted_mode", False),
            }
        except Exception as exc:
            yield None, {"error": str(exc)}

    @staticmethod
    def _stream_answer_chunks(answer: str) -> Iterator[str]:
        if not answer:
            return
        for line in answer.splitlines(keepends=True):
            if line:
                yield line

    def _record_dialogue_turn(
        self,
        *,
        question: str,
        result: dict[str, Any],
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
    ) -> None:
        if not session_id or enable_multi_turn:
            return

        metadata = {
            "intent": result.get("intent"),
            "repo_filter": repo_filter,
            "multi_turn": False,
            "graph_ready": result.get("graph_ready"),
            "restricted_mode": result.get("restricted_mode"),
            "searched_repositories": result.get("searched_repositories", []),
        }
        self._cache_manager.save_dialogue_turn(
            session_id=session_id,
            turn_number=self._get_next_turn_number(session_id),
            query=question,
            answer=result.get("answer", ""),
            summary=self._build_dialogue_summary(result),
            retrieved_elements=result.get("sources", []),
            metadata=metadata,
        )

    @staticmethod
    def _build_dialogue_summary(result: dict[str, Any]) -> str:
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources", [])
        if not sources:
            return answer[:500]

        lines = []
        for source in sources[:5]:
            repository = source.get("repository") or ""
            file_path = source.get("file") or ""
            location = "/".join(part for part in [repository, file_path] if part)
            if location:
                lines.append(f"- {location}")

        answer_preview = answer[:300]
        if lines:
            return "Files Read:\n" + "\n".join(lines) + f"\n\nAnswer Preview: {answer_preview}"
        return answer_preview

    def invalidate_index_scan_cache(self) -> None:
        self._index_scan_cache = None
        if self._legacy_runtime is not None:
            self._legacy_runtime.vector_store.invalidate_scan_cache()

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        if use_cache and self._index_scan_cache is not None:
            cache_time, cached = self._index_scan_cache
            if time.time() - cache_time < self._index_scan_cache_ttl:
                return cached

        results = [self._build_index_descriptor(root) for root in self._scan_index_roots()]
        results.sort(key=lambda repo: repo["name"])
        self._index_scan_cache = (time.time(), results)
        return results

    def _scan_index_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()

        repo_root = Path(getattr(self.loader, "safe_repo_root", self.config.get("repo_root", "./repos"))).resolve()
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
            }
        except Exception:
            return {
                "name": project_root.name,
                "element_count": 0,
                "file_count": 0,
                "size_mb": artifact_size_mb,
                "url": "N/A",
            }

    def _current_repository_descriptor(self) -> dict[str, Any]:
        element_count = 0
        if self.repo_indexed:
            container = self._ensure_container()
            element_count = len(container.ensure_graph().nodes)
        return {
            "name": self.repo_info.get("name", self.project_root.name if self.project_root else "Unknown"),
            "element_count": element_count,
            "file_count": self.repo_info.get("file_count", 0),
            "size_mb": self.repo_info.get("total_size_mb", 0),
            "url": self.repo_info.get("url", "N/A"),
        }

    def _sync_state_from_legacy(self) -> None:
        if self._legacy_runtime is None:
            return
        runtime = self._legacy_runtime
        self.repo_loaded = runtime.repo_loaded
        self.repo_indexed = runtime.repo_indexed
        self.repo_info = dict(runtime.repo_info)
        self.multi_repo_mode = runtime.multi_repo_mode
        self.loaded_repositories = dict(runtime.loaded_repositories)
        self.invalidate_index_scan_cache()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        runtime = self._ensure_legacy_runtime()
        return getattr(runtime, name)

    def _require_project_root(self) -> Path:
        if self.project_root is None:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        return self.project_root

    def _ensure_container(self) -> ServiceContainer:
        project_root = self._require_project_root()
        if self._container is None or self._container.project_root != project_root:
            self._container = ServiceContainer(project_root=project_root)
        return self._container

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
