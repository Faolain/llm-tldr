import json
from pathlib import Path


def _assert_profile_shape(path: Path):
    data = json.loads(path.read_text())

    assert data.get("schema_version") == 1
    assert data.get("suite_id") == "h2h_llm_tldr_vs_contextplus_v1"
    assert isinstance(data.get("tool_id"), str) and data.get("tool_id")
    assert isinstance(data.get("feature_set_id"), str) and data.get("feature_set_id")

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
    contextplus = repo_root / "benchmarks" / "head_to_head" / "tool_profiles" / "contextplus.v1.json"

    _assert_profile_shape(llm_tldr)
    _assert_profile_shape(contextplus)


def test_lane1_llm_tldr_profile_schema():
    repo_root = Path(__file__).resolve().parents[1]
    lane1 = (
        repo_root
        / "benchmarks"
        / "head_to_head"
        / "tool_profiles"
        / "llm_tldr.hybrid_lane1.v1.json"
    )

    _assert_profile_shape(lane1)


def test_contextplus_profile_is_real_profile_not_template():
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = repo_root / "benchmarks" / "head_to_head" / "tool_profiles" / "contextplus.v1.json"
    template_path = (
        repo_root
        / "benchmarks"
        / "head_to_head"
        / "tool_profiles"
        / "contextplus.v1.template.json"
    )

    assert profile_path.exists()
    assert template_path.exists()

    profile = json.loads(profile_path.read_text())
    assert profile.get("tool_id") == "contextplus"
    assert profile.get("capabilities", {}).get("retrieval") is True
    assert profile.get("capabilities", {}).get("impact") is False
    assert profile.get("capabilities", {}).get("slice") is False
    assert profile.get("capabilities", {}).get("complexity") is False
    assert profile.get("capabilities", {}).get("data_flow") is False


def test_contextplus_retrieval_template_has_no_placeholder_text():
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = repo_root / "benchmarks" / "head_to_head" / "tool_profiles" / "contextplus.v1.json"
    profile = json.loads(profile_path.read_text())

    retrieval = profile.get("commands", {}).get("retrieval", {})
    description = retrieval.get("description")
    template = retrieval.get("template")

    assert isinstance(description, str) and "Replace with the real" not in description
    assert isinstance(template, str) and template
    assert "{repo_root}" in template
    assert "{query}" in template
    assert "{top_k}" in template
