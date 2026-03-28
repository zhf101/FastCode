"""Pipeline-level graph validator.

Distinct from graph/validation.py (structural correctness).
This module checks pipeline *build-result completeness*: are there enough
nodes, are required fields present, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastcode.graph.models import GraphIssue, KnowledgeGraph

logger = logging.getLogger(__name__)

_MIN_NODES = 1


@dataclass
class PipelineValidationResult:
    valid: bool
    issues: list[GraphIssue] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.issues.append(GraphIssue(severity="warning", message=message))

    def add_error(self, message: str) -> None:
        self.issues.append(GraphIssue(severity="error", message=message))
        self.valid = False


def validate_pipeline_result(graph: KnowledgeGraph) -> PipelineValidationResult:
    """Check build-result completeness of *graph*.

    Returns a PipelineValidationResult. A non-valid result should block
    saving in strict mode but never crash the pipeline.
    """
    result = PipelineValidationResult(valid=True)

    if not graph.nodes:
        result.add_error("Graph contains no nodes — pipeline may have produced no output")
        return result

    if len(graph.nodes) < _MIN_NODES:
        result.add_warning(f"Graph has fewer than {_MIN_NODES} node(s); result may be incomplete")

    # Check that every edge references a real node
    node_ids = {n.id for n in graph.nodes}
    dangling = 0
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            dangling += 1
    if dangling:
        result.add_warning(f"{dangling} edge(s) reference unknown node IDs")

    # Warn if meta description is empty
    if not graph.project.description.strip():
        result.add_warning("ProjectMeta.description is empty")

    logger.info(
        "PipelineValidator: valid=%s, %d node(s), %d edge(s), %d issue(s)",
        result.valid,
        len(graph.nodes),
        len(graph.edges),
        len(result.issues),
    )
    return result
