"""Service container — holds shared graph services for a session."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastcode.graph.models import KnowledgeGraph
from fastcode.graph.persistence import graph_exists, load_graph
from fastcode.graph_services.query_context import QueryContextBuilder
from fastcode.app.intent_classifier import IntentClassifier


@dataclass
class ServiceContainer:
    """Lightweight DI container shared across a query session."""
    project_root: Path
    graph: KnowledgeGraph | None = None
    classifier: IntentClassifier = field(default_factory=IntentClassifier)
    context_builder: QueryContextBuilder = field(default_factory=QueryContextBuilder)
    _graph_loaded: bool = False

    # ------------------------------------------------------------------
    # Graph readiness
    # ------------------------------------------------------------------

    def graph_ready(self) -> bool:
        """Return True if a persisted graph exists for project_root."""
        return graph_exists(self.project_root)

    def ensure_graph(self) -> KnowledgeGraph:
        """Load and cache the graph. Raises FileNotFoundError if absent."""
        if not self._graph_loaded:
            if not self.graph_ready():
                raise FileNotFoundError(
                    f"No graph found under {self.project_root}/.fastcode/. "
                    "Run `build_graph` first."
                )
            self.graph = load_graph(self.project_root)
            self._graph_loaded = True
        return self.graph  # type: ignore[return-value]

    def invalidate_graph(self) -> None:
        """Force reload on next ensure_graph() call."""
        self.graph = None
        self._graph_loaded = False
