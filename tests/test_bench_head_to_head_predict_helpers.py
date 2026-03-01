import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_predict.py")


def test_render_command_template_raises_on_missing_placeholder():
    mod = _load_mod()
    render = mod["_render_command_template"]

    with pytest.raises(ValueError, match="missing template placeholders"):
        render("contextplus search --repo {repo_root} --query {query}", {"repo_root": "/tmp/repo"})


def test_timeout_maps_to_timeout_status_not_error():
    mod = _load_mod()
    run_once = mod["_run_command_once"]

    result = run_once(
        argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
        cwd=Path.cwd(),
        timeout_s=0.05,
    )

    assert result.status == "timeout"
    assert result.status != "error"


def test_raw_log_path_is_tool_trial_task_layout(tmp_path: Path):
    mod = _load_mod()
    raw_log_path = mod["_raw_log_path"]

    path = raw_log_path(
        tmp_path,
        tool_id="llm-tldr",
        trial=2,
        task_id="retrieval:R01",
    )

    expected = tmp_path / "benchmark" / "runs" / "raw_logs" / "llm-tldr" / "2" / "retrieval:R01.log"
    assert path == expected
