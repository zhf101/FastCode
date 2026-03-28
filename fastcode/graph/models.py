"""Core data models for the FastCode knowledge graph.

All graph consumers share this single contract. Do not introduce
alternate node/edge formats in other modules.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations (Literal types)
# ---------------------------------------------------------------------------

NodeType = Literal["file", "function", "class", "module", "concept"]

EdgeType = Literal[
    "contains",
    "imports",
    "calls",
    "inherits",
    "depends_on",
    "related",
]

Direction = Literal["directed", "undirected"]

Complexity = Literal["low", "moderate", "high"]

NodeSource = Literal["static", "llm", "hybrid"]

EdgeSource = Literal["static", "llm", "hybrid"]

AnalysisMode = Literal["full", "incremental"]

SymbolBackendType = Literal["ast", "serena_mcp", "hybrid"]

RuntimeMode = Literal["full", "restricted"]

# ---------------------------------------------------------------------------
# ProjectMeta
# ---------------------------------------------------------------------------


class ProjectMeta(BaseModel):
    name: str
    languages: list[str]
    frameworks: list[str]
    description: str
    entry_points: list[str] = Field(default_factory=list)

    @field_validator("languages", "frameworks", mode="before")
    @classmethod
    def _deduplicate(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    type: NodeType
    name: str
    file_path: str | None = None
    line_range: tuple[int, int] | None = None
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    complexity: Complexity | None = None
    language_notes: str | None = None
    source: NodeSource = "static"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary", mode="before")
    @classmethod
    def _default_summary(cls, v: str | None, info: Any) -> str:
        if not v:
            # Will be replaced by name via model_validator if still empty
            return ""
        return v

    @model_validator(mode="after")
    def _fill_summary(self) -> "GraphNode":
        if not self.summary:
            self.summary = self.name
        return self

    @field_validator("line_range", mode="before")
    @classmethod
    def _validate_line_range(cls, v: Any) -> Any:
        if v is None:
            return None
        start, end = v
        if start > end:
            raise ValueError(f"line_range start ({start}) must be <= end ({end})")
        return (start, end)


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------


class GraphEdge(BaseModel):
    source: str
    target: str
    type: EdgeType
    direction: Direction = "directed"
    weight: float = 0.5
    source_kind: EdgeSource = "static"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _no_self_loop(self) -> "GraphEdge":
        if self.source == self.target:
            raise ValueError(
                f"Self-loop not allowed: source and target are both '{self.source}'"
            )
        return self


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class Layer(BaseModel):
    id: str
    name: str
    description: str
    node_ids: list[str]

    @field_validator("node_ids")
    @classmethod
    def _no_duplicate_node_ids(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("node_ids must not contain duplicates")
        return v


# ---------------------------------------------------------------------------
# TourStep
# ---------------------------------------------------------------------------


class TourStep(BaseModel):
    order: int
    title: str
    description: str
    node_ids: list[str]
    language_lesson: str | None = None

    @field_validator("order")
    @classmethod
    def _order_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"order must be >= 1, got {v}")
        return v

    @field_validator("title")
    @classmethod
    def _title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v

    @field_validator("node_ids")
    @classmethod
    def _node_ids_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("node_ids must not be empty")
        return v


# ---------------------------------------------------------------------------
# KnowledgeGraph (top-level)
# ---------------------------------------------------------------------------


class KnowledgeGraph(BaseModel):
    version: str
    project: ProjectMeta
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    layers: list[Layer] = Field(default_factory=list)
    tour: list[TourStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AnalysisMeta  (stored separately in .fastcode/meta.json)
# ---------------------------------------------------------------------------


class AnalysisMeta(BaseModel):
    graph_version: str
    backend_version: str
    serena_version: str | None = None
    last_analyzed_at: str
    git_commit_hash: str
    analyzed_files: int
    changed_files: list[str] = Field(default_factory=list)
    analysis_mode: AnalysisMode
    symbol_backend: SymbolBackendType
    serena_available: bool
    runtime_mode: RuntimeMode
    warnings_count: int = 0

    @model_validator(mode="after")
    def _runtime_mode_consistency(self) -> "AnalysisMeta":
        if not self.serena_available and self.runtime_mode != "restricted":
            raise ValueError(
                "runtime_mode must be 'restricted' when serena_available is False"
            )
        return self


# ---------------------------------------------------------------------------
# GraphIssue  (stored separately in .fastcode/issues.json)
# ---------------------------------------------------------------------------

IssueSeverity = Literal["info", "warning", "error", "fatal"]


class GraphIssue(BaseModel):
    severity: IssueSeverity
    message: str
    node_id: str | None = None
    edge_source: str | None = None
    edge_target: str | None = None
    auto_fixed: bool = False
