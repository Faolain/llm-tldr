import json
from pathlib import Path


def _assert_profile_shape(path: Path):
    data = json.loads(path.read_text())

    assert data.get("schema_version") == 1
    assert data.get("suite_id") == "h2h_llm_tldr_vs_contextplus_v1"
    assert isinstance(data.get("tool_id"), str) and data.get("tool_id")

    caps = data.get("capabilities")
    assert isinstance(caps, dict)
    for key in ("retrieval", "impact", "slice", "complexity", "data_flow"):
        assert isinstance(caps.get(key), bool)

    commands = data.get("commands")
    assert isinstance(commands, dict)
    for key, supported in caps.items():
        if not supported:
            continue
        cmd = commands.get(key)
        assert isinstance(cmd, dict), f"missing command for supported capability {key}"
        assert isinstance(cmd.get("template"), str) and cmd.get("template")


def test_head_to_head_tool_profiles_schema():
    repo_root = Path(__file__).resolve().parents[1]

    llm_tldr = repo_root / "benchmarks" / "head_to_head" / "tool_profiles" / "llm_tldr.v1.json"
    contextplus = (
        repo_root
        / "benchmarks"
        / "head_to_head"
        / "tool_profiles"
        / "contextplus.v1.template.json"
    )

    _assert_profile_shape(llm_tldr)
    _assert_profile_shape(contextplus)
