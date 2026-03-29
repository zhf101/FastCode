"""Query orchestration helpers for GraphFirstFacade."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator


class GraphFirstQuerySupportMixin:
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
        del filters
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")

        options = self._query_options(
            repo_filter=repo_filter,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
            use_agency_mode=use_agency_mode,
            prompt_builder=prompt_builder,
        )
        if self._should_short_circuit_query(repo_filter=repo_filter):
            return self._empty_query_result(question)

        return self._dispatch_query(
            question,
            repo_filter=repo_filter,
            session_id=session_id,
            prompt_builder=prompt_builder,
            **options,
        )

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
        del filters
        options = self._query_options(
            repo_filter=repo_filter,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
            use_agency_mode=use_agency_mode,
            prompt_builder=prompt_builder,
        )
        if self._should_short_circuit_query(repo_filter=repo_filter):
            yield None, self._empty_query_result(question)
            return

        if self.multi_repo_mode:
            yield from self._query_stream_multi_repo(
                question,
                session_id=session_id,
                enable_multi_turn=options["enable_multi_turn"],
                repo_filter=repo_filter,
                prompt_builder=prompt_builder,
                use_agency_mode=options["use_agency_mode"],
            )
            return

        yield from self._query_stream_graph_first(
            question,
            session_id=session_id,
            enable_multi_turn=options["enable_multi_turn"],
            repo_filter=repo_filter,
            prompt_builder=prompt_builder,
            use_agency_mode=options["use_agency_mode"],
        )

    @staticmethod
    def _query_options(
        *,
        repo_filter: list[str] | None,
        session_id: str | None,
        enable_multi_turn: bool | None,
        use_agency_mode: bool | None,
        prompt_builder: Callable[..., str] | None,
    ) -> dict[str, Any]:
        del repo_filter, session_id, prompt_builder
        return {
            "enable_multi_turn": bool(enable_multi_turn),
            "use_agency_mode": use_agency_mode is True,
        }

    def _dispatch_query(
        self,
        question: str,
        *,
        repo_filter: list[str] | None,
        session_id: str | None,
        enable_multi_turn: bool,
        use_agency_mode: bool,
        prompt_builder: Callable[..., str] | None,
    ) -> dict[str, Any]:
        routed_question = self._prepare_graph_first_question(
            question,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
        )

        if self.multi_repo_mode:
            result = self._run_multi_repo_query(
                routed_question,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
            )
        else:
            result = (
                self._run_query_router_agency(routed_question)
                if use_agency_mode
                else self._run_query_router(routed_question)
            )

        return self._finalize_query_result(
            question=question,
            result=result,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
            repo_filter=repo_filter,
            prompt_builder=prompt_builder,
        )

    def _finalize_query_result(
        self,
        *,
        question: str,
        result: dict[str, Any],
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
        prompt_builder: Callable[..., str] | None,
    ) -> dict[str, Any]:
        result["query"] = question
        if prompt_builder is not None:
            result = self._apply_prompt_builder_compat(
                question=question,
                result=result,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
                prompt_builder=prompt_builder,
            )
        self._record_dialogue_turn(
            question=question,
            result=result,
            session_id=session_id,
            enable_multi_turn=enable_multi_turn,
            repo_filter=repo_filter,
        )
        return result

    def _should_short_circuit_query(
        self,
        *,
        repo_filter: list[str] | None,
    ) -> bool:
        if self.multi_repo_mode:
            return repo_filter is not None and not set(repo_filter).intersection(set(self.loaded_repositories))
        if repo_filter is None:
            return False
        current_repo = self.repo_info.get("name")
        return current_repo not in set(repo_filter)

    @staticmethod
    def _empty_query_result(question: str) -> dict[str, Any]:
        return {
            "answer": "No repositories are loaded for this query.",
            "query": question,
            "context_elements": 0,
            "sources": [],
            "searched_repositories": [],
            "graph_ready": False,
            "restricted_mode": False,
            "intent": "graph_qa",
            "intent_confidence": 0.0,
        }

    def _query_stream_multi_repo(
        self,
        question: str,
        *,
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
        prompt_builder: Callable[..., str] | None,
        use_agency_mode: bool,
    ) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
        try:
            yield None, {"status": "retrieving", "query": question}
            routed_question = self._prepare_graph_first_question(
                question,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
            )
            result = self._run_multi_repo_query(
                routed_question,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
            )
            result["query"] = question
            yield None, {
                "status": "generating",
                "sources": result.get("sources", []),
                "context_elements": result.get("context_elements", 0),
                "query": question,
            }

            if prompt_builder is not None:
                result = yield from self._stream_prompt_builder_compat(
                    question=question,
                    result=result,
                    session_id=session_id,
                    enable_multi_turn=enable_multi_turn,
                    repo_filter=repo_filter,
                    prompt_builder=prompt_builder,
                )
            else:
                answer = result.get("answer", "")
                for chunk in self._stream_answer_chunks(answer):
                    yield chunk, None

            answer = result.get("answer", "")

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
                "retrieval_available": result.get("retrieval_available"),
                "retrieval_unavailable_reason": result.get("retrieval_unavailable_reason"),
                "retrieval_backend_metadata": result.get("retrieval_backend_metadata"),
            }
        except Exception as exc:
            yield None, {"error": str(exc)}

    def _query_stream_graph_first(
        self,
        question: str,
        *,
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
        prompt_builder: Callable[..., str] | None,
        use_agency_mode: bool,
    ) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
        if not self.repo_loaded:
            yield None, {"error": "No repository loaded. Call load_repository() first."}
            return
        if not self.repo_indexed:
            yield None, {"error": "Repository not indexed. Call index_repository() first."}
            return

        try:
            yield None, {"status": "retrieving", "query": question}

            routed_question = self._prepare_graph_first_question(
                question,
                session_id=session_id,
                enable_multi_turn=enable_multi_turn,
            )
            if use_agency_mode:
                result = self._run_query_router_agency(routed_question)
            else:
                result = self._run_query_router(routed_question)
            result["query"] = question

            yield None, {
                "status": "generating",
                "sources": result.get("sources", []),
                "context_elements": result.get("context_elements", 0),
                "query": question,
            }

            if prompt_builder is not None:
                result = yield from self._stream_prompt_builder_compat(
                    question=question,
                    result=result,
                    session_id=session_id,
                    enable_multi_turn=enable_multi_turn,
                    repo_filter=repo_filter,
                    prompt_builder=prompt_builder,
                )
            else:
                answer = result.get("answer", "")
                for chunk in self._stream_answer_chunks(answer):
                    yield chunk, None

            answer = result.get("answer", "")

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
                "retrieval_available": result.get("retrieval_available"),
                "retrieval_unavailable_reason": result.get("retrieval_unavailable_reason"),
                "retrieval_backend_metadata": result.get("retrieval_backend_metadata"),
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
        if not session_id:
            return

        metadata = {
            "intent": result.get("intent"),
            "repo_filter": repo_filter,
            "multi_turn": enable_multi_turn,
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

    def _apply_prompt_builder_compat(
        self,
        *,
        question: str,
        result: dict[str, Any],
        session_id: str | None,
        enable_multi_turn: bool,
        prompt_builder: Callable[..., str],
    ) -> dict[str, Any]:
        generator = self._ensure_prompt_builder_answer_generator()
        generator.enable_multi_turn = enable_multi_turn
        generated = generator.generate_from_context(
            question,
            self._build_prompt_builder_context(result),
            query_info=self._build_prompt_builder_query_info(result),
            dialogue_history=self._get_prompt_builder_history(session_id, enable_multi_turn),
            prompt_builder=prompt_builder,
        )
        merged = dict(result)
        merged["answer"] = generated.get("answer", result.get("answer", ""))
        merged["query"] = question
        if "summary" in generated:
            merged["summary"] = generated["summary"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in generated:
                merged[key] = generated[key]
        return merged

    def _stream_prompt_builder_compat(
        self,
        *,
        question: str,
        result: dict[str, Any],
        session_id: str | None,
        enable_multi_turn: bool,
        repo_filter: list[str] | None,
        prompt_builder: Callable[..., str],
    ) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
        generator = self._ensure_prompt_builder_answer_generator()
        generator.enable_multi_turn = enable_multi_turn

        final_result = dict(result)
        answer_chunks: list[str] = []
        final_metadata: dict[str, Any] = {}

        for chunk, metadata in generator.generate_stream_from_context(
            question,
            self._build_prompt_builder_context(result),
            query_info=self._build_prompt_builder_query_info(result),
            dialogue_history=self._get_prompt_builder_history(session_id, enable_multi_turn),
            prompt_builder=prompt_builder,
        ):
            if chunk:
                answer_chunks.append(chunk)
                yield chunk, None
            if metadata:
                final_metadata.update(metadata)

        final_result["answer"] = "".join(answer_chunks)
        final_result["query"] = question
        if "summary" in final_metadata:
            final_result["summary"] = final_metadata["summary"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in final_metadata:
                final_result[key] = final_metadata[key]
        return final_result

    @staticmethod
    def _build_prompt_builder_context(result: dict[str, Any]) -> str:
        sections = ["## Graph-First Structured Answer", str(result.get("answer", ""))]
        sources = result.get("sources", [])
        if sources:
            lines = ["## Sources"]
            for source in sources:
                repository = source.get("repository") or source.get("repo") or ""
                file_path = source.get("file") or source.get("relative_path") or ""
                name = source.get("name") or "unknown"
                source_type = source.get("type") or "node"
                location = "/".join(part for part in [repository, file_path] if part)
                lines.append(f"- [{source_type}] {name}: {location}" if location else f"- [{source_type}] {name}")
            sections.append("\n".join(lines))
        return "\n\n".join(section for section in sections if section.strip())

    @staticmethod
    def _build_prompt_builder_query_info(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "intent": result.get("intent"),
            "graph_ready": result.get("graph_ready"),
            "restricted_mode": result.get("restricted_mode"),
            "context_elements": result.get("context_elements", 0),
            "searched_repositories": result.get("searched_repositories", []),
        }

    def _get_prompt_builder_history(
        self,
        session_id: str | None,
        enable_multi_turn: bool,
    ) -> list[dict[str, Any]] | None:
        if not enable_multi_turn or not session_id:
            return None
        history = self.get_session_history(session_id)
        return history or None

    def _prepare_graph_first_question(
        self,
        question: str,
        *,
        session_id: str | None,
        enable_multi_turn: bool,
    ) -> str:
        if not enable_multi_turn or not session_id:
            return question

        history = self._cache_manager.get_dialogue_history(session_id, max_turns=3)
        if not history:
            return question

        history_lines = []
        for turn in history:
            previous_query = str(turn.get("query", "")).strip()
            summary = str(turn.get("summary", "")).strip()
            if previous_query:
                history_lines.append(f"Q: {previous_query}")
            if summary:
                history_lines.append(f"S: {summary[:220]}")

        if not history_lines:
            return question

        return question + "\n\nConversation context:\n" + "\n".join(history_lines)

    def _run_multi_repo_query(
        self,
        question: str,
        *,
        repo_filter: list[str] | None,
        use_agency_mode: bool = False,
    ) -> dict[str, Any]:
        target_repositories = self._selected_multi_repo_entries(repo_filter)
        if not target_repositories:
            return self._empty_query_result(question)

        per_repo_results: list[tuple[str, dict[str, Any]]] = []
        total_context_elements = 0
        combined_sources: list[dict[str, Any]] = []
        confidence = 0.0

        for name, repo in target_repositories.items():
            project_root = Path(str(repo["root_path"]))
            if use_agency_mode:
                result = self._run_query_router_for_project_agency(question, project_root, repo_name=name)
            else:
                result = self._run_query_router_for_project(question, project_root, repo_name=name)
            per_repo_results.append((name, result))
            total_context_elements += int(result.get("context_elements", 0))
            combined_sources.extend(result.get("sources", []))
            confidence = max(confidence, float(result.get("intent_confidence", 0.0)))

        answer_parts = []
        for name, result in per_repo_results:
            answer_parts.append(f"## {name}\n")
            answer_parts.append(str(result.get("answer", "")))

        return {
            "answer": "\n\n".join(answer_parts),
            "query": question,
            "context_elements": total_context_elements,
            "sources": combined_sources,
            "searched_repositories": list(target_repositories),
            "graph_ready": True,
            "restricted_mode": any(bool(result.get("restricted_mode")) for _, result in per_repo_results),
            "intent": per_repo_results[0][1].get("intent", "graph_qa") if per_repo_results else "graph_qa",
            "intent_confidence": confidence,
        }

    def _selected_multi_repo_entries(
        self,
        repo_filter: list[str] | None,
    ) -> dict[str, dict[str, Any]]:
        if not self.multi_repo_mode:
            return {}
        if repo_filter is None:
            return dict(self.loaded_repositories)
        return {
            name: repo
            for name, repo in self.loaded_repositories.items()
            if name in repo_filter
        }
