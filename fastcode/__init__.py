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


__all__ = [
    "FastCode",
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

