"""
FastCode 2.0 - Repository-Level Code Understanding System
With Multi-Repository Support
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

__version__ = "2.0.0"

from .app.graph_first_facade import GraphFirstFacade


def _load_legacy():
    """Lazily import legacy modules to avoid mandatory heavy dependencies."""
    from .main import FastCode  # noqa: F401
    from .loader import RepositoryLoader  # noqa: F401
    from .parser import CodeParser  # noqa: F401
    from .indexer import CodeIndexer  # noqa: F401
    from .retriever import HybridRetriever  # noqa: F401
    from .answer_generator import AnswerGenerator  # noqa: F401
    from .repo_overview import RepositoryOverviewGenerator  # noqa: F401
    from .repo_selector import RepositorySelector  # noqa: F401
    from .iterative_agent import IterativeAgent  # noqa: F401
    from .agent_tools import AgentTools  # noqa: F401
    return (
        FastCode, RepositoryLoader, CodeParser, CodeIndexer, HybridRetriever,
        AnswerGenerator, RepositoryOverviewGenerator, RepositorySelector,
        IterativeAgent, AgentTools,
    )


def __getattr__(name: str):
    if name == "GraphFirstFacade":
        return GraphFirstFacade

    legacy_names = {
        "FastCode": 0,
        "RepositoryLoader": 1,
        "CodeParser": 2,
        "CodeIndexer": 3,
        "HybridRetriever": 4,
        "AnswerGenerator": 5,
        "RepositoryOverviewGenerator": 6,
        "RepositorySelector": 7,
        "IterativeAgent": 8,
        "AgentTools": 9,
    }
    if name in legacy_names:
        return _load_legacy()[legacy_names[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FastCode",
    "GraphFirstFacade",
    "RepositoryLoader",
    "CodeParser",
    "CodeIndexer",
    "HybridRetriever",
    "AnswerGenerator",
    "RepositoryOverviewGenerator",
    "RepositorySelector",
    "IterativeAgent",
    "AgentTools",
]

