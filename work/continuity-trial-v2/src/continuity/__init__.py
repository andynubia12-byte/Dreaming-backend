"""Continuity trial v2 — isolated staging of the personal-AI memory integration.

This package is a minimal, self-contained integration that wires the *existing*
8788 client (Pro main chat + Flash retrieval agent, DSML guard, new session) to
the *existing* long-term memory maintenance CLI (``memory-maintenance-v1``).

It runs entirely under ``work/continuity-trial-v2`` and never touches the live
8788 service, the other demos, the raw exports, or the production memory files.
"""

__all__ = [
    "config",
    "db",
    "util",
    "deepseek",
    "retrieval",
    "memory_agent",
    "agents",
    "summary_store",
    "context",
    "chat",
    "maintenance",
    "queue",
    "api",
    "history",
]
__version__ = "0.1.0"
