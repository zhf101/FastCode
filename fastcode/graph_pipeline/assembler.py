"""Assembler — converts SymbolIndex into GraphNodes and GraphEdges."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fastcode.graph.models import GraphEdge, GraphNode
from fastcode.symbol_backend.symbol_index import SymbolIndex
from fastcode.symbol_backend.protocol import SymbolInfo, RelationshipInfo

if TYPE_CHECKING:
    from .scanner import ScanResult

logger = logging.getLogger(__name__)

# Map provider kinds → GraphNode NodeType
_KIND_TO_NODE_TYPE: dict[str, str] = {
    "function": "function",
    "class": "class",
    "module": "module",
    "file": "file",
    "concept": "concept",
}

# Map relationship kinds → GraphEdge EdgeType
_REL_TO_EDGE_TYPE: dict[str, str] = {
    "imports": "imports",
    "calls": "calls",
    "inherits": "inherits",
    "contains": "contains",
    "depends_on": "depends_on",
    "related": "related",
}


def _node_id(file_path: str, name: str | None = None) -> str:
    """Stable, deterministic node ID from path + optional symbol name."""
    raw = f"{file_path}::{name}" if name else file_path
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class AssemblyResult:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    skipped_edges: int = 0


class Assembler:
    """Converts a SymbolIndex into graph nodes and edges."""

    def assemble(
        self,
        index: SymbolIndex,
        scan_result: "ScanResult | None" = None,
    ) -> AssemblyResult:
        """Build GraphNode/GraphEdge lists from *index*.

        Optionally uses *scan_result* to ensure every scanned file gets
        a file node even if the symbol provider returned nothing for it.
        """
        result = AssemblyResult()
        # Track emitted node IDs to avoid duplicates
        seen_node_ids: set[str] = set()

        # ------------------------------------------------------------------
        # 1. File nodes from scan_result (guaranteed coverage)
        # ------------------------------------------------------------------
        if scan_result is not None:
            for fp in scan_result.source_files:
                fp_str = str(fp)
                nid = _node_id(fp_str)
                if nid not in seen_node_ids:
                    result.nodes.append(GraphNode(
                        id=nid,
                        type="file",
                        name=fp.name,
                        file_path=fp_str,
                        source="static",
                    ))
                    seen_node_ids.add(nid)

        # ------------------------------------------------------------------
        # 2. Symbol nodes
        # ------------------------------------------------------------------
        for sym in index.all_symbols:
            node_type = _KIND_TO_NODE_TYPE.get(sym.kind, "function")
            nid = _node_id(sym.file_path, sym.name)
            if nid in seen_node_ids:
                continue
            seen_node_ids.add(nid)

            line_range: tuple[int, int] | None = None
            if sym.line_start or sym.line_end:
                start = sym.line_start or 0
                end = sym.line_end or start
                line_range = (start, end)

            result.nodes.append(GraphNode(
                id=nid,
                type=node_type,  # type: ignore[arg-type]
                name=sym.name,
                file_path=sym.file_path,
                line_range=line_range,
                summary=sym.docstring or "",
                source="static",
                metadata={"is_async": sym.is_async, "parent": sym.parent},
            ))

        # ------------------------------------------------------------------
        # 3. Contains edges: parent symbol → child symbol
        # ------------------------------------------------------------------
        for sym in index.all_symbols:
            if sym.parent:
                parent_id = _node_id(sym.file_path, sym.parent)
                child_id = _node_id(sym.file_path, sym.name)
                if parent_id in seen_node_ids and child_id in seen_node_ids:
                    result.edges.append(GraphEdge(
                        source=parent_id,
                        target=child_id,
                        type="contains",
                        source_type="static",
                    ))

        # ------------------------------------------------------------------
        # 4. Relationship edges
        # ------------------------------------------------------------------
        # Build a lookup: (file_path, name) -> node_id for target resolution
        name_to_ids: dict[str, list[str]] = {}
        for node in result.nodes:
            name_to_ids.setdefault(node.name, []).append(node.id)

        # file_path -> file node id
        file_to_id: dict[str, str] = {
            n.file_path: n.id
            for n in result.nodes
            if n.type == "file" and n.file_path
        }

        for rel in index.all_relationships:
            edge_type = _REL_TO_EDGE_TYPE.get(rel.kind, "related")

            # Resolve source node
            source_id = file_to_id.get(rel.source)
            if source_id is None:
                candidates = name_to_ids.get(rel.source)
                source_id = candidates[0] if candidates else None
            if source_id is None:
                result.skipped_edges += 1
                continue

            # Resolve target node
            target_id = file_to_id.get(rel.target)
            if target_id is None:
                candidates = name_to_ids.get(rel.target)
                target_id = candidates[0] if candidates else None
            if target_id is None:
                # Target may be an external module — skip silently
                result.skipped_edges += 1
                continue

            result.edges.append(GraphEdge(
                source=source_id,
                target=target_id,
                type=edge_type,  # type: ignore[arg-type]
                source_type="static",
            ))

        logger.info(
            "Assembler: %d nodes, %d edges, %d skipped edges",
            len(result.nodes),
            len(result.edges),
            result.skipped_edges,
        )
        return result
