"""Adapter objects used by GraphFirstFacade.

Keeps compatibility surfaces small and lets the facade focus on orchestration.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastcode.graph.persistence import load_graph
from fastcode.retrieval_runtime.code_retriever import CodeRetriever

if TYPE_CHECKING:
    from .graph_first_facade import GraphFirstFacade


class GraphFirstAnswerFormatter:
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

        if result.get("retrieval_available") is False:
            reason = result.get("retrieval_unavailable_reason") or "retrieval unavailable"
            output.append(f"\n\n## Retrieval Status\nUnavailable: {reason}")

        backend_meta = result.get("retrieval_backend_metadata") or {}
        if isinstance(backend_meta, dict):
            last_reload = backend_meta.get("last_reload_result")
            if isinstance(last_reload, dict):
                requested = last_reload.get("requested_repos", [])
                loaded = last_reload.get("loaded_repo_count", 0)
                vector_count = last_reload.get("vector_count", 0)
                bm25_count = last_reload.get("bm25_element_count", 0)
                failed = last_reload.get("failed_repos", [])

                output.append(
                    "\n\n## Retrieval Scope\n"
                    f"Loaded repos: {loaded}/{len(requested)} | "
                    f"Vectors: {vector_count} | BM25 elements: {bm25_count}"
                )
                if failed:
                    output.append("Failed repos: " + ", ".join(str(repo) for repo in failed))

        if "prompt_tokens" in result:
            output.append(
                f"\n\n*Used {result['prompt_tokens']} prompt tokens, "
                f"{result.get('context_elements', 0)} code snippets*"
            )

        return "\n".join(output)


class GraphFirstVectorStoreProxy:
    """Expose lightweight graph-first index scanning without eager legacy init."""

    def __init__(self, facade: "GraphFirstFacade") -> None:
        self._facade = facade

    def scan_available_indexes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        return self._facade.scan_available_indexes(use_cache=use_cache)

    def invalidate_scan_cache(self) -> None:
        self._facade.invalidate_index_scan_cache()

    def get_repository_names(self) -> list[str]:
        if self._facade.multi_repo_mode:
            return sorted(self._facade.loaded_repositories)
        return [repo["name"] for repo in self._facade.scan_available_indexes(use_cache=True)]

    def get_count(self) -> int:
        if self._facade.multi_repo_mode:
            return sum(
                int(repo.get("element_count", 0))
                for repo in self._facade.loaded_repositories.values()
            )
        if not self._facade.repo_indexed:
            return 0
        container = self._facade._ensure_container()
        return len(container.ensure_graph().nodes)


class GraphFirstRetrieverProxy:
    """Expose a minimal retriever surface without eager legacy init."""

    accurate_agent = None
    iterative_agent = None
    enable_agency_mode = False

    def __init__(self, facade: "GraphFirstFacade") -> None:
        self._facade = facade

    @property
    def repo_root(self) -> str | None:
        project_root = self._facade.project_root
        return str(project_root) if project_root is not None else None

    def set_repo_root(self, repo_root: str) -> None:
        if repo_root:
            self._facade.config["repo_root"] = repo_root

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        repo_filter: list[str] | None = None,
        enable_file_selection: bool = True,
        use_agency_mode: bool | None = None,
        dialogue_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del filters, enable_file_selection, dialogue_history
        return self._facade._retrieve_proxy_results(
            query,
            repo_filter=repo_filter,
            use_agency_mode=bool(use_agency_mode),
        )


class FallbackCacheManager:
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


class GraphFirstGraphElement:
    """Minimal CodeElement-like view for graph-first call-chain tooling."""

    def __init__(
        self,
        *,
        element_id: str,
        name: str,
        element_type: str,
        relative_path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> None:
        self.id = element_id
        self.name = name
        self.type = element_type
        self.relative_path = relative_path
        self.start_line = start_line or 0
        self.end_line = end_line or 0


class GraphFirstGraphBuilderProxy:
    """Small graph-builder compatibility layer backed by persisted graphs."""

    def __init__(self, facade: "GraphFirstFacade") -> None:
        self._facade = facade
        self.element_by_name: dict[str, GraphFirstGraphElement] = {}
        self.element_by_id: dict[str, GraphFirstGraphElement] = {}
        self._callers: dict[str, list[str]] = {}
        self._callees: dict[str, list[str]] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self.element_by_name = {}
        self.element_by_id = {}
        self._callers = {}
        self._callees = {}

        for repo in self._facade._graph_builder_repo_entries():
            root_path = Path(repo["root_path"])
            graph = load_graph(root_path)

            for node in graph.nodes:
                relative_path = self._facade._relative_repo_path(root_path, node.file_path)
                start_line, end_line = (node.line_range or (None, None))
                element = GraphFirstGraphElement(
                    element_id=node.id,
                    name=node.name,
                    element_type=node.type,
                    relative_path=relative_path,
                    start_line=start_line,
                    end_line=end_line,
                )
                self.element_by_id[element.id] = element
                self.element_by_name.setdefault(element.name, element)

            for edge in graph.edges:
                if edge.type != "calls":
                    continue
                self._callees.setdefault(edge.source, []).append(edge.target)
                self._callers.setdefault(edge.target, []).append(edge.source)

    def get_callers(self, element_id: str) -> list[str]:
        return list(self._callers.get(element_id, []))

    def get_callees(self, element_id: str) -> list[str]:
        return list(self._callees.get(element_id, []))
