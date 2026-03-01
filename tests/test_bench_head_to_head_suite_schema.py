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


def test_head_to_head_suite_django_pin_matches_corpora_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = repo_root / "benchmarks" / "head_to_head" / "suite.v1.json"
    suite = json.loads(suite_path.read_text())

    dataset = suite.get("dataset")
    assert isinstance(dataset, dict)
    assert dataset.get("corpus_id") == "django"
    assert dataset.get("required_git_sha") == "c04a09ddb3bb1fe8157292fcd902b35cad9a5e10"
    assert dataset.get("required_ref") == "5.1.13"

    corpus_manifest_rel = dataset.get("corpus_manifest")
    assert isinstance(corpus_manifest_rel, str) and corpus_manifest_rel
    corpus_manifest_path = repo_root / corpus_manifest_rel
    manifest = json.loads(corpus_manifest_path.read_text())

    corpora = manifest.get("corpora")
    assert isinstance(corpora, list)
    django = next((c for c in corpora if isinstance(c, dict) and c.get("id") == "django"), None)
    assert isinstance(django, dict)
    assert django.get("pinned_sha") == dataset.get("required_git_sha")
    assert django.get("pinned_ref") == dataset.get("required_ref")
