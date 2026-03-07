import json
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def test_agentic_judge_config_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "agentic" / "judge_config.json"
    data = _load(path)

    assert data.get("schema_version") == 1
    assert data.get("judge_provider") == "claude_cli"
    assert data.get("judge_model") == "sonnet"
    assert data.get("judge_effort") == "medium"
    assert data.get("judge_temperature") == 0.0
    assert data.get("judge_max_tokens") == 800
    assert data.get("judge_retries") == 1
    assert data.get("enforce_json_schema") is True


def test_agentic_preflight_tasks_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "agentic" / "preflight_tasks.json"
    data = _load(path)

    assert data.get("schema_version") == 1
    assert data.get("repo") == "django"
    tasks = data.get("tasks")
    assert isinstance(tasks, list)
    assert len(tasks) == 10

    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    allowed_workflows = {
        "concept_lookup",
        "exact_symbol_definition",
        "line_level_debugging",
        "refactor_blast_radius",
        "repeated_queries",
    }

    for task in tasks:
        assert isinstance(task, dict)
        task_id = task.get("id")
        assert isinstance(task_id, str) and task_id
        assert task_id not in seen_ids
        seen_ids.add(task_id)

        workflow_class = task.get("workflow_class")
        assert workflow_class in allowed_workflows
        counts[workflow_class] = counts.get(workflow_class, 0) + 1

        assert isinstance(task.get("question"), str) and task.get("question").strip()
        assert isinstance(task.get("target_repo"), str) and task.get("target_repo").strip()
        assert isinstance(task.get("expected_first_tool"), str) and task.get("expected_first_tool").strip()
        expected_tool_set = task.get("expected_tool_set")
        assert isinstance(expected_tool_set, list) and expected_tool_set
        assert all(isinstance(tool, str) and tool for tool in expected_tool_set)
        forbidden_first_tool = task.get("forbidden_first_tool")
        assert isinstance(forbidden_first_tool, list)
        assert isinstance(task.get("max_allowed_dead_end_turns"), int)

    assert counts == {
        "concept_lookup": 2,
        "exact_symbol_definition": 2,
        "line_level_debugging": 2,
        "refactor_blast_radius": 2,
        "repeated_queries": 2,
    }


def test_agentic_tool_choice_tasks_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "agentic" / "tool_choice_tasks.json"
    data = _load(path)

    assert data.get("schema_version") == 1
    assert data.get("repo") == "django"
    tasks = data.get("tasks")
    assert isinstance(tasks, list)
    assert len(tasks) == 30

    seen_ids: set[str] = set()
    workflow_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    allowed_workflows = {
        "concept_lookup",
        "exact_symbol_definition",
        "line_level_debugging",
        "refactor_blast_radius",
        "repeated_queries",
    }
    allowed_buckets = {"mixed", "rg_wins", "tldrf_wins"}

    for task in tasks:
        assert isinstance(task, dict)
        task_id = task.get("id")
        assert isinstance(task_id, str) and task_id
        assert task_id not in seen_ids
        seen_ids.add(task_id)

        workflow_class = task.get("workflow_class")
        assert workflow_class in allowed_workflows
        workflow_counts[workflow_class] = workflow_counts.get(workflow_class, 0) + 1

        bucket = task.get("comparison_bucket")
        assert bucket in allowed_buckets
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        assert task.get("expected_winner_arm") in {"augmented", "baseline"}
        assert isinstance(task.get("question"), str) and task.get("question").strip()
        assert isinstance(task.get("target_repo"), str) and task.get("target_repo").strip()
        assert isinstance(task.get("expected_first_tool"), str) and task.get("expected_first_tool").strip()
        expected_tool_set = task.get("expected_tool_set")
        assert isinstance(expected_tool_set, list) and expected_tool_set
        assert all(isinstance(tool, str) and tool for tool in expected_tool_set)
        assert isinstance(task.get("max_allowed_dead_end_turns"), int)

    assert workflow_counts == {
        "concept_lookup": 6,
        "exact_symbol_definition": 6,
        "line_level_debugging": 6,
        "refactor_blast_radius": 6,
        "repeated_queries": 6,
    }
    assert bucket_counts == {
        "mixed": 6,
        "rg_wins": 6,
        "tldrf_wins": 18,
    }


def test_agentic_patch_tasks_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "agentic" / "patch_tasks.json"
    data = _load(path)

    assert data.get("schema_version") == 1
    tasks = data.get("tasks")
    assert isinstance(tasks, list)
    assert len(tasks) == 10

    seen_ids: set[str] = set()
    allowed_categories = {
        "debugging_fix",
        "multi_file_change_impact",
        "small_refactor",
        "test_selection",
        "tests_update",
    }

    for task in tasks:
        assert isinstance(task, dict)
        task_id = task.get("id")
        assert isinstance(task_id, str) and task_id
        assert task_id not in seen_ids
        seen_ids.add(task_id)

        assert task.get("category") in allowed_categories
        assert isinstance(task.get("repo"), str) and task.get("repo").strip()
        assert isinstance(task.get("issue_description"), str) and task.get("issue_description").strip()
        assert isinstance(task.get("hidden_test_command"), str) and task.get("hidden_test_command").strip()
        expected_test_result = task.get("expected_test_result")
        assert isinstance(expected_test_result, dict)
        assert expected_test_result.get("exit_code") == 0
        expected_changed_files = task.get("expected_changed_files")
        assert isinstance(expected_changed_files, list) and expected_changed_files
        assert all(isinstance(path, str) and path for path in expected_changed_files)
        assert isinstance(task.get("max_turns"), int) and task.get("max_turns") > 0
        assert isinstance(task.get("timeout_s"), int) and task.get("timeout_s") > 0


def test_agentic_swebench_subset_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "agentic" / "swebench_subset.json"
    data = _load(path)

    assert data.get("schema_version") == 1
    assert data.get("repo") == "django/django"
    tasks = data.get("tasks")
    assert isinstance(tasks, list)
    assert len(tasks) >= 1

    for task in tasks:
        assert isinstance(task, dict)
        assert isinstance(task.get("instance_id"), str) and task.get("instance_id").strip()
        assert isinstance(task.get("repo"), str) and task.get("repo").strip()
        assert isinstance(task.get("base_commit"), str) and task.get("base_commit").strip()
        assert isinstance(task.get("problem_statement"), str) and task.get("problem_statement").strip()
        assert isinstance(task.get("test_cmd"), str) and task.get("test_cmd").strip()
        assert isinstance(task.get("timeout_s"), int) and task.get("timeout_s") > 0
        assert isinstance(task.get("resolved"), bool)


def test_agentic_gate_files_schema():
    repo_root = Path(__file__).resolve().parents[1]
    gate_dir = repo_root / "benchmarks" / "agentic"
    gate_names = [
        "phase_a_gates.json",
        "preflight_gates.json",
        "phase_c_gates.json",
        "phase_d_gates.json",
        "phase_e_gates.json",
        "phase_f_gates.json",
    ]

    for name in gate_names:
        data = _load(gate_dir / name)
        assert data.get("schema_version") == 1
        assert isinstance(data.get("phase"), str) and data.get("phase")

    phase_c = _load(gate_dir / "phase_c_gates.json")
    category_expectations = phase_c.get("category_expectations")
    assert isinstance(category_expectations, dict)
    assert set(category_expectations.keys()) == {"mixed", "rg_wins", "tldrf_wins"}
    for cfg in category_expectations.values():
        assert isinstance(cfg, dict)
        assert cfg.get("expected_winner") in {"augmented", "baseline"}
        assert isinstance(cfg.get("min_tasks"), int) and cfg.get("min_tasks") >= 5
        assert isinstance(cfg.get("win_rate_min"), float)
