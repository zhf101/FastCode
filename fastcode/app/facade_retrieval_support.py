"""Retrieval helpers for GraphFirstFacade and its lightweight proxies."""
from __future__ import annotations

from typing import Any

from fastcode.retrieval_runtime.code_retriever import CodeRetriever


class GraphFirstRetrievalSupportMixin:
    def _retrieve_proxy_results(
        self,
        query: str,
        *,
        repo_filter: list[str] | None,
        use_agency_mode: bool,
    ) -> list[dict[str, Any]]:
        if self.multi_repo_mode:
            results: list[dict[str, Any]] = []
            target_repositories = self._selected_multi_repo_entries(repo_filter)
            for repo_name, repo in target_repositories.items():
                retriever = self._build_isolated_code_retriever(repo["root_path"])
                results.extend(
                    self._snippet_to_payload(snippet, repository=repo_name)
                    for snippet in self._retrieve_snippets(
                        retriever,
                        query,
                        use_agency_mode=use_agency_mode,
                        max_results=5,
                    )
                )
            return results

        project_root = self._require_project_root()
        retriever = self._get_cached_code_retriever(project_root)
        if retriever is None:
            return []

        repo_name = self.repo_info.get("name") if self.repo_info else None
        return [
            self._snippet_to_payload(snippet, repository=repo_name)
            for snippet in self._retrieve_snippets(
                retriever,
                query,
                use_agency_mode=use_agency_mode,
                max_results=5,
            )
        ]

    @staticmethod
    def _retrieve_snippets(
        retriever: CodeRetriever,
        query: str,
        *,
        use_agency_mode: bool,
        max_results: int,
    ):
        if use_agency_mode:
            from fastcode.retrieval_runtime.iterative_retriever import IterativeRetriever

            iterative = IterativeRetriever(retriever)
            return iterative.retrieve(query, max_results=max_results).snippets
        return retriever.retrieve(query, max_results=max_results).snippets

    @staticmethod
    def _snippet_to_payload(snippet: Any, *, repository: str | None) -> dict[str, Any]:
        payload = {
            "file_path": snippet.file_path,
            "content": snippet.content,
            "score": snippet.score,
            "line_start": snippet.line_start,
            "line_end": snippet.line_end,
            "name": snippet.symbol_name,
            "node_id": snippet.node_id,
        }
        if repository:
            payload["repository"] = repository
        return payload
