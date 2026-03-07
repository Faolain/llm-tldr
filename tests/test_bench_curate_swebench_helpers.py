import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_curate_swebench.py")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_curate_subset_filters_resolved_repo_and_test_cmd():
    mod = _load_mod()
    curate_subset = mod["curate_subset"]

    rows = [
        {
            "instance_id": "django-2",
            "repo": "django/django",
            "resolved": True,
            "base_commit": "abc",
            "patch": "diff --git",
            "test_cmd": "python -m pytest tests/test_two.py",
        },
        {
            "instance_id": "django-1",
            "repo": "django/django",
            "resolved": True,
            "base_commit": "def",
            "patch": "diff --git",
            "test_command": "python -m pytest tests/test_one.py",
        },
        {
            "instance_id": "django-open",
            "repo": "django/django",
            "resolved": False,
            "base_commit": "ghi",
            "patch": "diff --git",
            "test_cmd": "python -m pytest tests/test_open.py",
        },
        {
            "instance_id": "requests-1",
            "repo": "psf/requests",
            "resolved": True,
            "base_commit": "zzz",
            "patch": "diff --git",
            "test_cmd": "python -m pytest tests/test_other.py",
        },
        {
            "instance_id": "django-missing-cmd",
            "repo": "django/django",
            "resolved": True,
            "base_commit": "yyy",
            "patch": "diff --git",
        },
    ]

    curated = curate_subset(rows, repo="django/django", count=10, default_timeout_s=120)

    assert [row["instance_id"] for row in curated] == ["django-1", "django-2"]
    assert curated[0]["test_cmd"] == "python -m pytest tests/test_one.py"
    assert curated[0]["timeout_s"] == 120


def test_read_source_supports_jsonl(tmp_path: Path):
    mod = _load_mod()
    read_source = mod["_read_source"]

    source = tmp_path / "swebench.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "a", "repo": "django/django"}),
                json.dumps({"instance_id": "b", "repo": "django/django"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_source(source)
    assert [row["instance_id"] for row in rows] == ["a", "b"]


def test_main_writes_subset_doc(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    source = tmp_path / "verified.json"
    out = tmp_path / "subset.json"
    _write_json(
        source,
        [
            {
                "instance_id": "django-2",
                "repo": "django/django",
                "resolved": True,
                "base_commit": "abc",
                "patch": "diff --git",
                "test_cmd": "python -m pytest tests/test_two.py",
            },
            {
                "instance_id": "django-1",
                "repo": "django/django",
                "resolved": True,
                "base_commit": "def",
                "patch": "diff --git",
                "test_cmd": "python -m pytest tests/test_one.py",
            },
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_curate_swebench.py",
            "--source",
            str(source),
            "--count",
            "1",
            "--out",
            str(out),
        ],
    )

    assert mod["main"]() == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["repo"] == "django/django"
    assert doc["count"] == 1
    assert doc["tasks"][0]["instance_id"] == "django-1"
