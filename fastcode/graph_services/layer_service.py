"""Layer service — heuristic layer detection from graph nodes."""
from __future__ import annotations

import hashlib
import re

from fastcode.graph.models import GraphNode, KnowledgeGraph, Layer

# ---------------------------------------------------------------------------
# Layer heuristics
# ---------------------------------------------------------------------------

# Ordered: first match wins for a given node
_LAYER_RULES: list[tuple[str, list[str]]] = [
    ("Test",    ["test", "tests", "spec", "specs"]),
    ("Config",  ["config", "conf", "settings", "setup"]),
    ("UI",      ["ui", "frontend", "web", "static", "templates", "views"]),
    ("API",     ["api", "routes", "endpoints", "controllers", "handlers"]),
    ("Service", ["service", "services", "application", "app", "use_cases"]),
    ("Data",    ["model", "models", "schema", "schemas", "db", "database",
                 "repository", "repositories", "orm", "migration", "migrations"]),
    ("Utility", ["util", "utils", "helper", "helpers", "common", "shared",
                 "lib", "tools"]),
]

_LAYER_FALLBACK = "Core"


def _layer_for_path(file_path: str) -> str:
    """Map a file path to a layer name by matching path segments."""
    segments = re.split(r"[/\\]", file_path.lower())
    for layer_name, keywords in _LAYER_RULES:
        for seg in segments:
            if any(kw == seg or seg.startswith(kw) for kw in keywords):
                return layer_name
    return _LAYER_FALLBACK


def _layer_id(name: str) -> str:
    return "layer:" + hashlib.sha1(name.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_layers(graph: KnowledgeGraph) -> list[Layer]:
    """Detect layers by grouping nodes via path-pattern heuristics.

    Every node with a file_path is assigned to exactly one layer;
    nodes without file_path are grouped into Core.
    """
    bucket: dict[str, list[str]] = {}

    for node in graph.nodes:
        fp = node.file_path or ""
        layer_name = _layer_for_path(fp) if fp else _LAYER_FALLBACK
        bucket.setdefault(layer_name, []).append(node.id)

    layers: list[Layer] = []
    for name, node_ids in sorted(bucket.items()):
        layers.append(Layer(
            id=_layer_id(name),
            name=name,
            description=f"{name} layer (auto-detected)",
            node_ids=list(dict.fromkeys(node_ids)),  # deduplicate, preserve order
        ))
    return layers


def apply_llm_layers(
    graph: KnowledgeGraph,
    llm_layers: list[dict],
) -> list[Layer]:
    """Merge LLM-supplied layer overrides onto heuristic layers.

    LLM may supply dicts with keys: name, description. Node assignment
    is NOT changed — only description text may be overridden.
    """
    heuristic = {l.name: l for l in detect_layers(graph)}
    for override in llm_layers:
        name = override.get("name", "")
        desc = override.get("description", "")
        if name in heuristic and desc:
            layer = heuristic[name]
            heuristic[name] = layer.model_copy(update={"description": desc})
    return list(heuristic.values())
