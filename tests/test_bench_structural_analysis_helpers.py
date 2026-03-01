import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_structural_analysis.py")


def test_data_flow_origin_accuracy_requires_exact_origin_line():
    mod = _load_mod()
    data_flow_eval = mod["_data_flow_eval"]

    dfg = {
        "refs": [
            {"name": "total", "line": 10},
            {"name": "total", "line": 12},
        ],
        "edges": [
            {"var": "total", "def_line": 10, "use_line": 12},
        ],
    }

    out = data_flow_eval(dfg, variable="total", expected_lines={10, 12})
    assert out["origin_line"] == 10
    assert out["origin_line"] == 10
    assert out["origin_line"] != 11


def test_python_function_span_resolves_class_method_names():
    mod = _load_mod()
    function_span = mod["_python_function_span"]

    source = "\n".join(
        [
            "class Service:",
            "    def run(self):",
            "        value = 1",
            "        return value",
            "",
            "def top_level():",
            "    return 0",
        ]
    )
    assert function_span(source, function="Service.run") == (2, 4)
