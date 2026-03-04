"""Tests for daemon impact response normalization.

The daemon's ``_handle_impact`` handler returns caller entries with the key
``"caller"`` (e.g. ``{"caller": "check_dependencies", "file": "main.py", "line": 12}``),
while the benchmark parser and downstream reports use the key ``"function"``.

The normalisation bridge lives in ``_daemon_response_to_stdout("impact", ...)``,
which rewrites ``"caller"`` -> ``"function"`` before the JSON is fed to the
parsing pipeline.  These tests verify:

1. The normalization itself (``_daemon_response_to_stdout``).
2. Edge cases: empty callers, callers that already have ``"function"``, mixed
   callers, non-ok status passthrough.
3. Round-trip: ``_daemon_response_to_stdout`` output piped through the
   downstream ``_parse_impact_result`` parser produces correct results.
"""

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_predict.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mod():
    """Load the benchmark module once per test module."""
    return _load_mod()


@pytest.fixture()
def daemon_response_to_stdout(mod):
    return mod["_daemon_response_to_stdout"]


@pytest.fixture()
def parse_impact_result(mod):
    return mod["_parse_impact_result"]


@pytest.fixture()
def extract_json(mod):
    return mod["_extract_json_from_text"]


# ---------------------------------------------------------------------------
# _daemon_response_to_stdout: basic normalization
# ---------------------------------------------------------------------------

class TestDaemonResponseToStdoutImpactNormalization:
    """Verify that _daemon_response_to_stdout normalises 'caller' -> 'function'."""

    def test_single_caller_key_renamed(self, daemon_response_to_stdout):
        """The daemon's 'caller' key should produce only 'function' in output."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "check_dependencies", "file": "utils.py", "line": 42},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out["callers"][0]["function"] == "check_dependencies"
        assert "caller" not in out["callers"][0]

    def test_no_input_mutation(self, daemon_response_to_stdout):
        """Normalisation should not mutate the input response object."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "check_dependencies", "file": "utils.py", "line": 42},
            ],
        }
        original = copy.deepcopy(daemon_resp)
        _ = daemon_response_to_stdout("impact", daemon_resp)
        assert daemon_resp == original

    def test_multiple_callers_all_get_function_key(self, daemon_response_to_stdout):
        """Every entry in the callers list should get a 'function' key."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "main", "file": "main.py", "line": 1},
                {"caller": "run_tests", "file": "tests.py", "line": 55},
                {"caller": "deploy", "file": "deploy.py", "line": None},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        for entry in out["callers"]:
            assert "function" in entry
        assert [e["function"] for e in out["callers"]] == ["main", "run_tests", "deploy"]

    def test_caller_already_has_function_key_not_overwritten(self, daemon_response_to_stdout):
        """If an entry already has 'function', it should NOT be overwritten by 'caller'."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "old_name", "function": "correct_name", "file": "a.py", "line": 1},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        # The fix checks: if "caller" in entry AND "function" not in entry.
        # When "function" already exists, it should be preserved unchanged.
        assert out["callers"][0]["function"] == "correct_name"

    def test_mixed_callers_only_missing_function_normalised(self, daemon_response_to_stdout):
        """Mix of entries: some with 'caller' only, some with 'function' already."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "needs_rename", "file": "a.py", "line": 1},
                {"function": "already_correct", "file": "b.py", "line": 2},
                {"caller": "also_needs_rename", "file": "c.py", "line": 3},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        funcs = [e.get("function") for e in out["callers"]]
        assert funcs == ["needs_rename", "already_correct", "also_needs_rename"]

    def test_empty_callers_list(self, daemon_response_to_stdout):
        """Empty callers list should pass through without error."""
        daemon_resp = {
            "status": "ok",
            "callers": [],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out["callers"] == []

    def test_no_callers_key(self, daemon_response_to_stdout):
        """Response with no callers key at all should pass through."""
        daemon_resp = {"status": "ok"}
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out["status"] == "ok"

    def test_callers_not_a_list(self, daemon_response_to_stdout):
        """If callers is not a list, pass through unchanged."""
        daemon_resp = {"status": "ok", "callers": "unexpected_string"}
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out["callers"] == "unexpected_string"

    def test_non_dict_entries_in_callers_preserved(self, daemon_response_to_stdout):
        """Non-dict entries in the callers list should be preserved as-is."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                "not_a_dict",
                {"caller": "valid_entry", "file": "a.py", "line": 1},
                42,
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out["callers"][0] == "not_a_dict"
        assert out["callers"][1]["function"] == "valid_entry"
        assert out["callers"][2] == 42

    def test_file_and_line_preserved(self, daemon_response_to_stdout):
        """The 'file' and 'line' keys should be preserved after normalisation."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "process", "file": "src/handler.py", "line": 99},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        entry = out["callers"][0]
        assert entry["file"] == "src/handler.py"
        assert entry["line"] == 99

    def test_result_key_preserved(self, daemon_response_to_stdout):
        """The 'result' key (impact_analysis tree) should pass through unchanged."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "main", "file": "main.py", "line": 1},
            ],
            "result": {
                "targets": {
                    "helper@utils.py": {
                        "function": "helper",
                        "file": "utils.py",
                        "caller_count": 1,
                        "callers": [
                            {"function": "main", "file": "main.py", "caller_count": 0, "callers": []},
                        ],
                    }
                },
                "total_targets": 1,
            },
        }
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        # Top-level callers normalised
        assert out["callers"][0]["function"] == "main"
        # Nested result.targets tree should be untouched (already uses 'function')
        target = out["result"]["targets"]["helper@utils.py"]
        assert target["function"] == "helper"
        assert target["callers"][0]["function"] == "main"


# ---------------------------------------------------------------------------
# _daemon_response_to_stdout: status passthrough
# ---------------------------------------------------------------------------

class TestDaemonResponseToStdoutStatusPassthrough:
    """Non-ok status responses should pass through without normalisation."""

    def test_error_status_passthrough(self, daemon_response_to_stdout):
        daemon_resp = {"status": "error", "message": "Missing required parameter: func"}
        out = json.loads(daemon_response_to_stdout("impact", daemon_resp))
        assert out == daemon_resp

    def test_non_impact_category_not_affected(self, daemon_response_to_stdout):
        """Normalisation only applies to the 'impact' category."""
        daemon_resp = {
            "status": "ok",
            "results": [{"caller": "should_stay", "file": "a.py"}],
        }
        out = json.loads(daemon_response_to_stdout("retrieval", daemon_resp))
        # Retrieval responses should NOT rename 'caller' keys
        assert out["results"][0]["caller"] == "should_stay"


# ---------------------------------------------------------------------------
# Round-trip: _daemon_response_to_stdout -> _parse_impact_result
# ---------------------------------------------------------------------------

class TestRoundTripDaemonToParsed:
    """End-to-end: daemon response -> normalisation -> parser -> final result."""

    def test_roundtrip_legacy_shape(self, daemon_response_to_stdout, parse_impact_result, extract_json):
        """Legacy daemon shape: callers with 'caller' key, no result tree."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "main", "file": "main.py", "line": 12},
                {"caller": "test_helper", "file": "tests/test_main.py", "line": 5},
            ],
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        assert result["callers"] == [
            {"file": "main.py", "function": "main"},
            {"file": "tests/test_main.py", "function": "test_helper"},
        ]

    def test_parser_accepts_caller_key_directly(self, parse_impact_result):
        """Parser should accept legacy caller entries even without normalisation."""
        raw_daemon_data = {
            "callers": [
                {"caller": "main", "file": "main.py", "line": 12},
                {"caller": "test_it", "file": "test.py", "line": 5},
            ],
        }
        result = parse_impact_result(raw_daemon_data)
        assert result["callers"] == [
            {"file": "main.py", "function": "main"},
            {"file": "test.py", "function": "test_it"},
        ]

    def test_roundtrip_newer_shape_with_result_tree(
        self, daemon_response_to_stdout, parse_impact_result, extract_json
    ):
        """Newer daemon shape: top-level callers + result.targets tree."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "entry_point", "file": "src/app.py", "line": 10},
            ],
            "result": {
                "targets": {
                    "helper@src/utils.py": {
                        "function": "helper",
                        "file": "src/utils.py",
                        "caller_count": 1,
                        "callers": [
                            {
                                "function": "entry_point",
                                "file": "src/app.py",
                                "caller_count": 0,
                                "callers": [],
                                "truncated": False,
                            },
                        ],
                        "truncated": False,
                    }
                },
                "total_targets": 1,
            },
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        # The parser extracts from both top-level callers AND result.targets
        # and deduplicates.  "entry_point" appears in both, so only once.
        assert len(result["callers"]) == 1
        assert result["callers"][0] == {"file": "src/app.py", "function": "entry_point"}

    def test_roundtrip_empty_callers(self, daemon_response_to_stdout, parse_impact_result, extract_json):
        """Empty callers from daemon should yield empty callers from parser."""
        daemon_resp = {
            "status": "ok",
            "callers": [],
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        assert result["callers"] == []

    def test_roundtrip_deduplication(self, daemon_response_to_stdout, parse_impact_result, extract_json):
        """Duplicate callers from top-level and nested tree should be deduped."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "run", "file": "run.py", "line": 1},
                {"caller": "run", "file": "run.py", "line": 1},
            ],
            "result": {
                "targets": {
                    "do_work@work.py": {
                        "function": "do_work",
                        "file": "work.py",
                        "caller_count": 1,
                        "callers": [
                            {
                                "function": "run",
                                "file": "run.py",
                                "caller_count": 0,
                                "callers": [],
                                "truncated": False,
                            },
                        ],
                        "truncated": False,
                    }
                },
                "total_targets": 1,
            },
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        # "run" at "run.py" appears 3 times total (2 top-level + 1 nested),
        # but should be deduped to just 1.
        assert len(result["callers"]) == 1
        assert result["callers"][0] == {"file": "run.py", "function": "run"}

    def test_roundtrip_path_normalisation(
        self, daemon_response_to_stdout, parse_impact_result, extract_json
    ):
        """Paths with './' prefix should be normalised by the parser."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "init", "file": "./src/init.py", "line": 3},
            ],
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        assert result["callers"][0]["file"] == "src/init.py"

    def test_roundtrip_error_yields_empty_callers(
        self, daemon_response_to_stdout, parse_impact_result, extract_json
    ):
        """An error response from the daemon should yield empty callers from parser."""
        daemon_resp = {"status": "error", "message": "call graph not loaded"}
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)

        assert result["callers"] == []


# ---------------------------------------------------------------------------
# Parser compatibility: legacy caller key
# ---------------------------------------------------------------------------

class TestParserCallerKeyCompatibility:
    """Verify parser compatibility with both 'caller' and 'function' keys."""

    def test_raw_daemon_callers_parsed_by_parser(self, parse_impact_result):
        """Raw daemon caller entries should parse directly via fallback lookup."""
        raw_daemon_data = {
            "callers": [
                {"caller": "main", "file": "main.py", "line": 12},
                {"caller": "test_it", "file": "test.py", "line": 5},
            ],
        }
        result = parse_impact_result(raw_daemon_data)
        assert result["callers"] == [
            {"file": "main.py", "function": "main"},
            {"file": "test.py", "function": "test_it"},
        ]

    def test_normalised_daemon_callers_parsed_correctly(
        self, daemon_response_to_stdout, parse_impact_result, extract_json
    ):
        """After normalisation, the same data parses correctly."""
        daemon_resp = {
            "status": "ok",
            "callers": [
                {"caller": "main", "file": "main.py", "line": 12},
                {"caller": "test_it", "file": "test.py", "line": 5},
            ],
        }
        stdout_text = daemon_response_to_stdout("impact", daemon_resp)
        parsed = extract_json(stdout_text)
        result = parse_impact_result(parsed)
        assert len(result["callers"]) == 2
        assert result["callers"][0] == {"file": "main.py", "function": "main"}
        assert result["callers"][1] == {"file": "test.py", "function": "test_it"}


# ---------------------------------------------------------------------------
# Daemon response schema verification
# ---------------------------------------------------------------------------

class TestDaemonResponseSchemaDocumentation:
    """Verify the assumed daemon response schema shapes used by the normaliser."""

    def test_legacy_shape_matches_daemon_handler(self, daemon_response_to_stdout):
        """Legacy shape: {status, callers: [{caller, file, line}]}."""
        legacy = {
            "status": "ok",
            "callers": [
                {"caller": "main", "file": "main.py", "line": 12},
            ],
        }
        out = json.loads(daemon_response_to_stdout("impact", legacy))
        assert out["status"] == "ok"
        assert out["callers"][0]["function"] == "main"

    def test_newer_shape_matches_daemon_handler(self, daemon_response_to_stdout):
        """Newer shape: {status, callers: [{caller, file, line}], result: {targets: ...}, meta: ...}."""
        newer = {
            "status": "ok",
            "callers": [
                {"caller": "entry", "file": "app.py", "line": 1},
            ],
            "result": {
                "targets": {},
                "total_targets": 0,
            },
            "meta": {"language": "python"},
        }
        out = json.loads(daemon_response_to_stdout("impact", newer))
        assert out["status"] == "ok"
        assert out["callers"][0]["function"] == "entry"
        assert out["result"] == {"targets": {}, "total_targets": 0}
        assert out["meta"] == {"language": "python"}
