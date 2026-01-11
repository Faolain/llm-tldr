"""Session stats tracking for TLDR token savings.

Tracks per-session token usage to show value proposition:
- Raw tokens (what vanilla Claude would have used)
- TLDR tokens (what was actually returned)
- Savings percentage

Also tracks per-hook activity metrics:
- Hook invocations and success rates
- Hook-specific metrics (errors found, queries routed, etc.)

Stats are persisted to JSONL for historical analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken

# Lazy-loaded encoder (singleton)
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get or create tiktoken encoder (lazy loading)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Uses cl100k_base encoding (same as GPT-4/Claude).

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    encoder = _get_encoder()
    return len(encoder.encode(text))


@dataclass
class SessionStats:
    """Stats for a single session.

    Accumulates token counts across multiple requests.
    """

    session_id: str
    raw_tokens: int = 0
    tldr_tokens: int = 0
    requests: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record_request(self, raw_tokens: int, tldr_tokens: int) -> None:
        """Record a request's token usage.

        Args:
            raw_tokens: Tokens that would have been used for raw file
            tldr_tokens: Tokens actually used by TLDR response
        """
        self.raw_tokens += raw_tokens
        self.tldr_tokens += tldr_tokens
        self.requests += 1

    @property
    def savings_tokens(self) -> int:
        """Total tokens saved."""
        return self.raw_tokens - self.tldr_tokens

    @property
    def savings_percent(self) -> float:
        """Savings as percentage (0-100)."""
        if self.raw_tokens == 0:
            return 0.0
        return (self.savings_tokens / self.raw_tokens) * 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "session_id": self.session_id,
            "raw_tokens": self.raw_tokens,
            "tldr_tokens": self.tldr_tokens,
            "requests": self.requests,
            "savings_tokens": self.savings_tokens,
            "savings_percent": round(self.savings_percent, 2),
            "timestamp": datetime.now(UTC).isoformat(),
            "started_at": self.started_at.isoformat(),
        }


@dataclass
class HookStats:
    """Stats for a single hook type.

    Tracks invocations and hook-specific metrics.
    """

    hook_name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    metrics: dict[str, int | float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record_invocation(self, success: bool = True, metrics: dict | None = None) -> None:
        """Record a hook invocation.

        Args:
            success: Whether the hook succeeded
            metrics: Hook-specific metrics to accumulate
        """
        self.invocations += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # Accumulate metrics
        if metrics:
            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key] += value
                else:
                    self.metrics[key] = value

    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100)."""
        if self.invocations == 0:
            return 100.0
        return (self.successes / self.invocations) * 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "hook_name": self.hook_name,
            "invocations": self.invocations,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 2),
            "metrics": self.metrics,
            "started_at": self.started_at.isoformat(),
        }


class StatsStore:
    """JSONL-based stats persistence.

    Stores session stats in append-only JSONL format for durability
    and easy querying.
    """

    def __init__(self, path: Path | str):
        """Initialize stats store.

        Args:
            path: Path to JSONL file
        """
        self.path = Path(path)

    def append(self, stats: SessionStats) -> None:
        """Append session stats to JSONL file.

        Args:
            stats: Session stats to persist
        """
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Append as single line
        with open(self.path, "a") as f:
            f.write(json.dumps(stats.to_dict()) + "\n")

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get all records for a specific session.

        Args:
            session_id: Session ID to filter by

        Returns:
            List of stats records for this session
        """
        if not self.path.exists():
            return []

        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("session_id") == session_id:
                        records.append(record)
                except json.JSONDecodeError:
                    continue

        return records

    def get_totals(self) -> dict[str, int]:
        """Get all-time totals across all sessions.

        Returns:
            Dict with raw_tokens, tldr_tokens, requests totals
        """
        if not self.path.exists():
            return {"raw_tokens": 0, "tldr_tokens": 0, "requests": 0}

        totals = {"raw_tokens": 0, "tldr_tokens": 0, "requests": 0}

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    totals["raw_tokens"] += record.get("raw_tokens", 0)
                    totals["tldr_tokens"] += record.get("tldr_tokens", 0)
                    totals["requests"] += record.get("requests", 0)
                except json.JSONDecodeError:
                    continue

        return totals

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most recent records.

        Args:
            limit: Max number of records to return

        Returns:
            List of recent stats records
        """
        if not self.path.exists():
            return []

        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Return last N records
        return records[-limit:]


# Default store location
def get_default_store() -> StatsStore:
    """Get the default stats store.

    Location: ~/.cache/tldr/session_stats.jsonl
    """
    cache_dir = Path.home() / ".cache" / "tldr"
    return StatsStore(cache_dir / "session_stats.jsonl")


class HookStatsStore:
    """Project-local hook stats persistence.

    Stores hook activity stats in JSONL format within the project's .tldr directory.
    Multiple Claude instances can write to the same file - stats are aggregated on load.

    Location: {project}/.tldr/stats/hook_activity.jsonl
    """

    def __init__(self, project_path: Path | str):
        """Initialize hook stats store for a project.

        Args:
            project_path: Root path of the project
        """
        self.project = Path(project_path)
        self.stats_dir = self.project / ".tldr" / "stats"
        self.path = self.stats_dir / "hook_activity.jsonl"

    def load(self) -> dict[str, HookStats]:
        """Load and aggregate hook stats from JSONL file.

        Returns:
            Dict mapping hook names to aggregated HookStats
        """
        if not self.path.exists():
            return {}

        aggregated: dict[str, HookStats] = {}

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    hook_name = record.get("hook_name")
                    if not hook_name:
                        continue

                    if hook_name not in aggregated:
                        aggregated[hook_name] = HookStats(hook_name=hook_name)

                    # Aggregate invocations
                    stats = aggregated[hook_name]
                    stats.invocations += record.get("invocations", 0)
                    stats.successes += record.get("successes", 0)
                    stats.failures += record.get("failures", 0)

                    # Aggregate metrics
                    for key, value in record.get("metrics", {}).items():
                        if key in stats.metrics:
                            stats.metrics[key] += value
                        else:
                            stats.metrics[key] = value

                except json.JSONDecodeError:
                    # Skip corrupted lines
                    continue

        return aggregated

    def append(self, hook_stats: dict[str, HookStats]) -> None:
        """Append current hook stats snapshot to JSONL.

        Args:
            hook_stats: Dict of hook name -> HookStats to persist
        """
        if not hook_stats:
            return

        # Ensure directory exists
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # Append each hook's current stats as a record
        timestamp = datetime.now(UTC).isoformat()
        with open(self.path, "a") as f:
            for stats in hook_stats.values():
                if stats.invocations > 0:
                    record = {
                        "hook_name": stats.hook_name,
                        "invocations": stats.invocations,
                        "successes": stats.successes,
                        "failures": stats.failures,
                        "metrics": stats.metrics,
                        "timestamp": timestamp,
                    }
                    f.write(json.dumps(record) + "\n")

    def flush_delta(self, current: dict[str, HookStats], baseline: dict[str, HookStats]) -> None:
        """Flush only the delta (new activity) since baseline.

        This allows multiple instances to append without double-counting.

        Args:
            current: Current in-memory stats
            baseline: Stats at last flush (or load time)
        """
        if not current:
            return

        self.stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).isoformat()
        with open(self.path, "a") as f:
            for hook_name, stats in current.items():
                base = baseline.get(hook_name)
                if base:
                    # Calculate delta
                    delta_invocations = stats.invocations - base.invocations
                    delta_successes = stats.successes - base.successes
                    delta_failures = stats.failures - base.failures
                    delta_metrics = {}
                    for key, value in stats.metrics.items():
                        base_val = base.metrics.get(key, 0)
                        delta_metrics[key] = value - base_val
                else:
                    # No baseline - write all
                    delta_invocations = stats.invocations
                    delta_successes = stats.successes
                    delta_failures = stats.failures
                    delta_metrics = stats.metrics

                # Only write if there's actual activity
                if delta_invocations > 0:
                    record = {
                        "hook_name": hook_name,
                        "invocations": delta_invocations,
                        "successes": delta_successes,
                        "failures": delta_failures,
                        "metrics": delta_metrics,
                        "timestamp": timestamp,
                    }
                    f.write(json.dumps(record) + "\n")
