"""
TLDR Daemon package.

Provides the TLDRDaemon server and lifecycle management functions.
"""

from .cached_queries import (
    cached_architecture,
    cached_cfg,
    cached_context,
    cached_dead_code,
    cached_dfg,
    cached_extract,
    cached_importers,
    cached_imports,
    cached_search,
    cached_slice,
    cached_structure,
    cached_tree,
)
from .core import IDLE_TIMEOUT, TLDRDaemon
from .startup import (
    main,
    query_daemon,
    start_daemon,
    stop_daemon,
)

__all__ = [
    # Core
    "TLDRDaemon",
    "IDLE_TIMEOUT",
    # Lifecycle
    "start_daemon",
    "stop_daemon",
    "query_daemon",
    "main",
    # Cached queries
    "cached_search",
    "cached_extract",
    "cached_dead_code",
    "cached_architecture",
    "cached_cfg",
    "cached_dfg",
    "cached_slice",
    "cached_tree",
    "cached_structure",
    "cached_context",
    "cached_imports",
    "cached_importers",
]
