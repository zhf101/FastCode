"""Derived artifacts orchestrator — generates layers and tour and writes them back to the graph."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastcode.graph.models import KnowledgeGraph
from fastcode.graph.persistence import save_graph
from fastcode.graph.validation import validate_graph
from fastcode.graph_services.layer_service import detect_layers
from fastcode.graph_services.tour_service import generate_heuristic_tour

logger = logging.getLogger(__name__)


@dataclass
class ArtifactOptions:
    generate_layers: bool = True
    generate_tour: bool = True
    persist: bool = True


@dataclass
class ArtifactResult:
    graph: KnowledgeGraph
    layers_generated: int = 0
    tour_steps_generated: int = 0
    validation_valid: bool = True
    warnings: list[str] = field(default_factory=list)


def build_derived_artifacts(
    graph: KnowledgeGraph,
    project_root: Path,
    options: ArtifactOptions | None = None,
) -> ArtifactResult:
    """Generate layers and tour, write them back to *graph*, optionally persist.

    Args:
        graph:        The KnowledgeGraph to enrich.
        project_root: Project root directory (for persistence).
        options:      Control which artifacts to generate and whether to save.

    Returns:
        ArtifactResult with the updated graph and generation stats.
    """
    if options is None:
        options = ArtifactOptions()

    updates: dict = {}
    layers_n = 0
    tour_n = 0
    warnings: list[str] = []

    # 1. Layers
    if options.generate_layers:
        layers = detect_layers(graph)
        updates["layers"] = layers
        layers_n = len(layers)
        logger.info("build_derived_artifacts: generated %d layer(s)", layers_n)

    # 2. Tour (uses layers if just generated)
    if options.generate_tour:
        # Use updated layers for tour generation if available
        graph_for_tour = graph.model_copy(update=updates) if updates else graph
        tour = generate_heuristic_tour(graph_for_tour)
        updates["tour"] = tour
        tour_n = len(tour)
        logger.info("build_derived_artifacts: generated %d tour step(s)", tour_n)

    # 3. Apply updates
    updated_graph = graph.model_copy(update=updates) if updates else graph

    # 4. Validate
    val = validate_graph(updated_graph)
    if not val.valid:
        for issue in val.issues:
            warnings.append(f"[{issue.severity}] {issue.message}")
            logger.warning("ArtifactsValidator: %s", issue.message)

    # 5. Persist
    if options.persist:
        save_graph(updated_graph, project_root, strict=False)
        logger.info("build_derived_artifacts: graph persisted to %s/.fastcode/", project_root)

    return ArtifactResult(
        graph=updated_graph,
        layers_generated=layers_n,
        tour_steps_generated=tour_n,
        validation_valid=val.valid,
        warnings=warnings,
    )
