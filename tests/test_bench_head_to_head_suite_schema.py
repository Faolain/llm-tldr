import json
from pathlib import Path


def test_head_to_head_suite_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "head_to_head" / "suite.v1.json"
    data = json.loads(path.read_text())

    assert data.get("schema_version") == 1
    assert data.get("suite_id") == "h2h_llm_tldr_vs_contextplus_v1"

    dataset = data.get("dataset")
    assert isinstance(dataset, dict)
    assert dataset.get("corpus_id") == "django"
    assert isinstance(dataset.get("required_git_sha"), str)
    assert isinstance(dataset.get("required_ref"), str)

    sources = data.get("sources")
    assert isinstance(sources, dict)
    assert isinstance(sources.get("retrieval_queries"), str)
    assert isinstance(sources.get("structural_queries"), str)

    lanes = data.get("lanes")
    assert isinstance(lanes, list)
    assert lanes

    lane_ids = set()
    has_required_retrieval = False
    for lane in lanes:
        assert isinstance(lane, dict)
        lid = lane.get("id")
        assert isinstance(lid, str) and lid
        assert lid not in lane_ids
        lane_ids.add(lid)

        cats = lane.get("categories")
        assert isinstance(cats, list) and cats
        assert all(isinstance(c, str) for c in cats)

        if bool(lane.get("required_for_all_tools")) and "retrieval" in cats:
            has_required_retrieval = True

    assert has_required_retrieval, "suite must require retrieval for all tools"

    budgets = data.get("budgets")
    assert isinstance(budgets, dict)
    token_budgets = budgets.get("token_budgets")
    assert isinstance(token_budgets, list)
    assert token_budgets == sorted(token_budgets)
    assert 2000 in token_budgets

    protocol = data.get("protocol")
    assert isinstance(protocol, dict)
    assert isinstance(protocol.get("trials"), int)
    seeds = protocol.get("seeds")
    assert isinstance(seeds, list)
    assert len(seeds) == protocol.get("trials")

    gates = data.get("gates")
    assert isinstance(gates, dict)
    assert isinstance(gates.get("run_validity"), dict)
    assert isinstance(gates.get("tool_quality"), dict)
    assert isinstance(gates.get("head_to_head"), dict)
