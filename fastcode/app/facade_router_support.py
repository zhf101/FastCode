"""Router execution helpers for GraphFirstFacade."""
from __future__ import annotations

from pathlib import Path

from fastcode.graph.persistence import graph_exists
from fastcode.retrieval_runtime.code_retriever import CodeRetriever

from .query_router import QueryRouter
from .service_container import ServiceContainer


class GraphFirstRouterSupportMixin:
    def _run_query_router(self, question: str) -> dict[str, object]:
        project_root = self._require_project_root()
        return self._run_cached_router_query(question, project_root=project_root)

    def _run_query_router_agency(self, question: str) -> dict[str, object]:
        project_root = self._require_project_root()
        return self._run_cached_router_query(
            question,
            project_root=project_root,
            use_agency_mode=True,
        )

    def _run_query_router_for_project(
        self,
        question: str,
        project_root: Path,
        *,
        repo_name: str | None = None,
    ) -> dict[str, object]:
        return self._run_isolated_router_query(
            question,
            project_root=project_root,
            repo_name=repo_name,
        )

    def _run_query_router_for_project_agency(
        self,
        question: str,
        project_root: Path,
        *,
        repo_name: str | None = None,
    ) -> dict[str, object]:
        return self._run_isolated_router_query(
            question,
            project_root=project_root,
            repo_name=repo_name,
            use_agency_mode=True,
        )

    def _run_cached_router_query(
        self,
        question: str,
        *,
        project_root: Path,
        use_agency_mode: bool = False,
    ) -> dict[str, object]:
        self._ensure_router(project_root)
        container = self._ensure_container()
        result = self.router.route(question, container, use_agency_mode=use_agency_mode)
        self._container = container
        repo_name = self.repo_info.get("name") if self.repo_info else None
        return self._build_router_payload(result, question=question, repo_name=repo_name)

    def _run_isolated_router_query(
        self,
        question: str,
        *,
        project_root: Path,
        repo_name: str | None = None,
        use_agency_mode: bool = False,
    ) -> dict[str, object]:
        router = QueryRouter(code_retriever=self._build_isolated_code_retriever(project_root))
        container = ServiceContainer(project_root=project_root)
        result = router.route(question, container, use_agency_mode=use_agency_mode)
        return self._build_router_payload(result, question=question, repo_name=repo_name)

    def _get_cached_code_retriever(self, project_root: Path) -> CodeRetriever | None:
        self._ensure_router(project_root)
        return self._code_retriever

    def _build_isolated_code_retriever(self, project_root: str | Path) -> CodeRetriever:
        return CodeRetriever.from_runtime_config(self.config, repo_root=project_root)

    def _ensure_router(self, project_root: Path) -> None:
        if self._code_retriever is not None:
            if self.router is None:
                self.router = QueryRouter(code_retriever=self._code_retriever)
            return
        if self.router is None:
            self.router = QueryRouter()
        if not graph_exists(project_root):
            return

        self._code_retriever = self._build_isolated_code_retriever(project_root)
        self.router = QueryRouter(code_retriever=self._code_retriever)

    @staticmethod
    def _build_router_payload(
        result,
        *,
        question: str,
        repo_name: str | None,
    ) -> dict[str, object]:
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
            "retrieval_available": result.retrieval_available,
            "retrieval_unavailable_reason": result.retrieval_unavailable_reason,
        }
