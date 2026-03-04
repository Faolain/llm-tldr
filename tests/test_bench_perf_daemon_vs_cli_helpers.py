import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_perf_daemon_vs_cli.py")


def test_speedup_metric_uses_p50_latency_for_gate_alignment():
    mod = _load_mod()
    speedup = mod["_speedup"]

    cli = {"stats_ms": {"mean": 120.0, "p50": 100.0}}
    daemon = {"stats_ms": {"mean": 20.0, "p50": 40.0}}

    assert speedup(cli, daemon) == 2.5
