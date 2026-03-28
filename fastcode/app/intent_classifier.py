"""Intent classifier — keyword-rule-based query intent detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    graph_qa = "graph_qa"      # architecture / structure / relationship questions
    explain = "explain"        # explain a file, class, function
    diff = "diff"              # impact of changes
    onboard = "onboard"        # new-member project orientation
    hybrid_detail = "hybrid_detail"  # requires both graph + code detail
    unknown = "unknown"


@dataclass
class ClassificationResult:
    intent: QueryIntent
    confidence: float          # 0.0 – 1.0
    matched_signals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal tables
# ---------------------------------------------------------------------------

_GRAPH_QA_SIGNALS = [
    "架构", "模块", "关系", "依赖", "结构", "调用", "层级", "入口", "流程",
    "architecture", "module", "dependency", "relationship", "structure",
    "entry point", "flow", "how does", "what is", "where is",
    "overview", "map", "graph", "组件", "component",
]

_EXPLAIN_SIGNALS = [
    "解释", "说明", "explain", "what does", "describe",
    "这个函数", "这个类", "this function", "this class", "this file",
]

_DIFF_SIGNALS = [
    "影响", "改动", "变化", "diff", "change", "impact", "affect",
    "修改", "会不会影响", "what if", "breaking",
]

_ONBOARD_SIGNALS = [
    "新成员", "如何理解", "onboard", "getting started", "new to",
    "how to start", "where do i begin", "入门", "从哪里开始",
]

_HYBRID_SIGNALS = [
    "实现细节", "implementation", "具体实现", "source code", "源码",
    "line by line", "逐行",
]

_SIGNAL_MAP: list[tuple[list[str], QueryIntent]] = [
    (_EXPLAIN_SIGNALS, QueryIntent.explain),
    (_DIFF_SIGNALS, QueryIntent.diff),
    (_ONBOARD_SIGNALS, QueryIntent.onboard),
    (_HYBRID_SIGNALS, QueryIntent.hybrid_detail),
    (_GRAPH_QA_SIGNALS, QueryIntent.graph_qa),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Rule-based intent classifier. No LLM dependency."""

    def classify(self, query: str) -> ClassificationResult:
        """Classify *query* into a QueryIntent.

        Matching is case-insensitive substring search across all signal tables.
        The intent with the most matched signals wins; ties break in table order.
        """
        q_lower = query.lower()
        scores: dict[QueryIntent, list[str]] = {intent: [] for _, intent in _SIGNAL_MAP}

        for signals, intent in _SIGNAL_MAP:
            for signal in signals:
                if signal.lower() in q_lower:
                    scores[intent].append(signal)

        best_intent = QueryIntent.unknown
        best_signals: list[str] = []
        for _, intent in _SIGNAL_MAP:
            if len(scores[intent]) > len(best_signals):
                best_intent = intent
                best_signals = scores[intent]

        if not best_signals:
            return ClassificationResult(
                intent=QueryIntent.unknown,
                confidence=0.0,
            )

        confidence = min(1.0, 0.5 + 0.1 * len(best_signals))
        return ClassificationResult(
            intent=best_intent,
            confidence=confidence,
            matched_signals=best_signals,
        )
