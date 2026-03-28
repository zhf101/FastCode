"""Session and prompt-builder helpers for GraphFirstFacade."""
from __future__ import annotations


class GraphFirstSessionSupportMixin:
    def list_sessions(self) -> list[dict[str, object]]:
        sessions = self._cache_manager.list_sessions()
        enriched_sessions: list[dict[str, object]] = []
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

    def get_session_history(self, session_id: str) -> list[dict[str, object]]:
        return self._cache_manager.get_dialogue_history(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._cache_manager.delete_session(session_id)

    def _get_next_turn_number(self, session_id: str) -> int:
        session_index = self._cache_manager._get_session_index(session_id)
        if session_index:
            return session_index.get("total_turns", 0) + 1
        return 1

    def _ensure_prompt_builder_answer_generator(self):
        if self._prompt_builder_answer_generator is None:
            from fastcode.answer_generator import AnswerGenerator

            self._prompt_builder_answer_generator = AnswerGenerator(self.config)
        return self._prompt_builder_answer_generator
