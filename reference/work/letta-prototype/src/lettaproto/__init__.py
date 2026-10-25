"""Letta real-data prototype (phase 1 backend, idea.md centered).

The user chose to reuse their existing dreaming output (idea.md) as the working
memory instead of re-mining all history. So the pipeline is:

  canonical chat index (for on-demand source lookup)
    + working memory idea.md (injected in full, versioned)
    + incremental updater (new exchanges / explicit corrections only)
    + query (idea first, then history with provenance)
    + Letta bridge (sync idea.md into a local agent) and a direct chat engine.

Small modules, no framework.
"""

__all__ = [
    "config", "util", "db", "importer", "deepseek", "ideastore", "query",
    "context", "chat", "updater", "letta", "api", "agents", "worker", "vendor",
]
__version__ = "0.3.0"
