#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_util import bench_runs_root, get_repo_root, write_report

SCHEMA_VERSION = 1
PREFLIGHT_SEMANTIC_INDEX_MISSING_CLASS = "preflight_semantic_index_missing"
PREFLIGHT_SEMANTIC_INDEX_MISSING_EXPLICIT_CLASSES = frozenset(
    {
        "preflight_semantic_index_missing",
        "preflight_missing_semantic_index",
        "semantic_index_missing_preflight",
    }
)
PREFLIGHT_SEMANTIC_INDEX_MISSING_REASON_MARKERS = ("semantic index not found",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json_obj(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _parse_filter_values(raw_values: list[Any] | None) -> list[str]:
    values: list[str] = []
    for raw in raw_values or []:
        for token in str(raw).split(","):
            item = token.strip()
            if item:
                values.append(item)
    return values


def _parse_positive_int_filter(raw_values: list[Any] | None, *, label: str) -> list[int]:
    out: list[int] = []
    for raw in _parse_filter_values(raw_values):
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f"{label} values must be integers: {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{label} values must be positive: {raw!r}")
        out.append(value)
    return sorted(set(out))


def _normalize_explicit_allowlist_filters(
    *,
    task_ids_raw: list[Any] | None,
    trials_raw: list[Any] | None,
    budget_tokens_raw: list[Any] | None,
) -> dict[str, list[Any]]:
    return {
        "task_ids": sorted(set(_parse_filter_values(task_ids_raw))),
        "trials": _parse_positive_int_filter(trials_raw, label="trial"),
        "budget_tokens": _parse_positive_int_filter(budget_tokens_raw, label="budget_tokens"),
    }


def _as_list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _coerce_explicit_allowlist_filters(filters: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not isinstance(filters, dict):
        return _normalize_explicit_allowlist_filters(
            task_ids_raw=None,
            trials_raw=None,
            budget_tokens_raw=None,
        )
    return _normalize_explicit_allowlist_filters(
        task_ids_raw=_as_list_or_none(filters.get("task_ids")),
        trials_raw=_as_list_or_none(filters.get("trials")),
        budget_tokens_raw=_as_list_or_none(filters.get("budget_tokens")),
    )


def _has_explicit_allowlist_filters(filters: dict[str, list[Any]]) -> bool:
    return any(filters.get(field) for field in ("task_ids", "trials", "budget_tokens"))


def _key_matches_explicit_allowlist(
    key: tuple[str, str, int, int], filters: dict[str, list[Any]]
) -> bool:
    _, task_id, budget, trial = key
    task_ids = {str(x) for x in filters.get("task_ids", []) if str(x)}
    trials = {int(x) for x in filters.get("trials", [])}
    budgets = {int(x) for x in filters.get("budget_tokens", [])}

    if task_ids and task_id not in task_ids:
        return False
    if trials and trial not in trials:
        return False
    if budgets and budget not in budgets:
        return False
    return True


def _normalize_row_key(tool_id: str, row: dict[str, Any]) -> tuple[str, str, int, int] | None:
    task_id = row.get("task_id")
    budget = row.get("budget_tokens")
    trial = row.get("trial")
    if not isinstance(task_id, str) or not isinstance(budget, int) or not isinstance(trial, int):
        return None
    return (tool_id, task_id, int(budget), int(trial))


def _classification_row_map(
    classification_doc: dict[str, Any], default_tool_id: str
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    rows = classification_doc.get("rows")
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tool_id = row.get("tool")
        if not isinstance(tool_id, str) or not tool_id:
            tool_id = default_tool_id
        key = _normalize_row_key(tool_id, row)
        if key is None:
            continue
        out[key] = row
    return out


def _classification_map(classification_doc: dict[str, Any], default_tool_id: str) -> dict[tuple[str, str, int, int], str]:
    out: dict[tuple[str, str, int, int], str] = {}
    for key, row in _classification_row_map(classification_doc, default_tool_id=default_tool_id).items():
        failure_class = row.get("failure_class")
        if isinstance(failure_class, str) and failure_class:
            out[key] = failure_class
    return out


def _failure_class_for_row(
    *,
    tool_id: str,
    row: dict[str, Any],
    class_map: dict[tuple[str, str, int, int], str],
    use_classification_map: bool = True,
) -> str:
    if use_classification_map:
        key = _normalize_row_key(tool_id, row)
        if key is not None and key in class_map:
            return class_map[key]

    status = row.get("status")
    if status == "timeout":
        return "provider_transport_runtime"
    if status == "error":
        return "product_failure"
    if status == "pending":
        return "unclassified"
    return "none"


def _is_preflight_semantic_index_missing(
    *,
    explicit_failure_class: str | None,
    status: str | None,
    reason: str | None,
) -> bool:
    explicit = str(explicit_failure_class or "").strip().lower()
    if explicit:
        return explicit in PREFLIGHT_SEMANTIC_INDEX_MISSING_EXPLICIT_CLASSES

    status_norm = str(status or "").strip().lower()
    if status_norm != "error":
        return False

    reason_norm = str(reason or "").strip().lower()
    return any(marker in reason_norm for marker in PREFLIGHT_SEMANTIC_INDEX_MISSING_REASON_MARKERS)


def _replacement_base_failure_class(
    *,
    tool_id: str,
    row: dict[str, Any],
    class_map: dict[tuple[str, str, int, int], str],
    class_row_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> str:
    failure_class = _failure_class_for_row(
        tool_id=tool_id,
        row=row,
        class_map=class_map,
        use_classification_map=True,
    )
    if failure_class == "provider_transport_runtime":
        return failure_class

    key = _normalize_row_key(tool_id, row)
    class_row = class_row_map.get(key) if key is not None else None

    explicit_failure_class: str | None = None
    if isinstance(class_row, dict):
        raw_class = class_row.get("failure_class")
        if isinstance(raw_class, str) and raw_class.strip():
            explicit_failure_class = raw_class
        status = class_row.get("status")
        reason = class_row.get("reason")
    else:
        status = row.get("status")
        reason = row.get("reason")

    if _is_preflight_semantic_index_missing(
        explicit_failure_class=explicit_failure_class,
        status=status if isinstance(status, str) else None,
        reason=reason if isinstance(reason, str) else None,
    ):
        return PREFLIGHT_SEMANTIC_INDEX_MISSING_CLASS

    return failure_class


def _key_to_dict(key: tuple[str, str, int, int]) -> dict[str, Any]:
    tool, task_id, budget, trial = key
    return {
        "tool": tool,
        "task_id": task_id,
        "budget": budget,
        "trial": trial,
    }


def _validate_identity(base: dict[str, Any], rerun: dict[str, Any], rerun_path: Path) -> None:
    required_equal = (
        "schema_version",
        "suite_id",
        "tool_id",
        "task_manifest_sha256",
    )
    for field in required_equal:
        if base.get(field) != rerun.get(field):
            raise ValueError(
                f"rerun identity mismatch for {field}: base={base.get(field)!r} rerun={rerun.get(field)!r}"
                f" ({rerun_path})"
            )


def _stitch_predictions(
    *,
    base_doc: dict[str, Any],
    rerun_docs: list[tuple[Path, dict[str, Any]]],
    classification_doc: dict[str, Any],
    run_metadata_doc: dict[str, Any] | None,
    base_path: Path,
    explicit_allowlist_filters: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tool_id = base_doc.get("tool_id")
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("base predictions missing tool_id")

    base_predictions = base_doc.get("predictions")
    if not isinstance(base_predictions, list):
        raise ValueError("base predictions missing list")

    if run_metadata_doc is not None:
        md_hash = run_metadata_doc.get("task_manifest_sha256")
        base_hash = base_doc.get("task_manifest_sha256")
        if isinstance(md_hash, str) and isinstance(base_hash, str) and md_hash != base_hash:
            raise ValueError("run metadata task_manifest_sha256 mismatch")

    for rerun_path, rerun_doc in rerun_docs:
        _validate_identity(base_doc, rerun_doc, rerun_path)

    class_row_map = _classification_row_map(classification_doc, default_tool_id=tool_id)
    class_map = _classification_map(classification_doc, default_tool_id=tool_id)
    allowlist_filters = _coerce_explicit_allowlist_filters(explicit_allowlist_filters)
    allowlist_enabled = _has_explicit_allowlist_filters(allowlist_filters)

    base_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    base_order: list[tuple[str, str, int, int]] = []
    for row in base_predictions:
        if not isinstance(row, dict):
            continue
        key = _normalize_row_key(tool_id, row)
        if key is None:
            continue
        if key in base_by_key:
            raise ValueError(f"duplicate base key: {key}")
        base_by_key[key] = row
        base_order.append(key)

    if allowlist_enabled:
        allowlist_matches = [key for key in base_order if _key_matches_explicit_allowlist(key, allowlist_filters)]
        if not allowlist_matches:
            raise ValueError("explicit stitch allowlist filters matched zero base rows")

    rerun_maps: list[tuple[Path, dict[tuple[str, str, int, int], dict[str, Any]]]] = []
    for rerun_path, rerun_doc in rerun_docs:
        rows = rerun_doc.get("predictions")
        if not isinstance(rows, list):
            raise ValueError(f"rerun predictions missing list: {rerun_path}")
        row_map: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = _normalize_row_key(tool_id, row)
            if key is None:
                continue
            row_map[key] = row
        rerun_maps.append((rerun_path, row_map))

    stitched_predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for key in base_order:
        base_row = base_by_key[key]
        base_class = _replacement_base_failure_class(
            tool_id=tool_id,
            row=base_row,
            class_map=class_map,
            class_row_map=class_row_map,
        )
        explicit_eligible = allowlist_enabled and _key_matches_explicit_allowlist(key, allowlist_filters)
        failure_eligible = base_class in {"provider_transport_runtime", PREFLIGHT_SEMANTIC_INDEX_MISSING_CLASS}
        if failure_eligible and explicit_eligible:
            eligibility_source = "failure_class_and_explicit_allowlist"
        elif failure_eligible:
            eligibility_source = "failure_class"
        elif explicit_eligible:
            eligibility_source = "explicit_allowlist"
        else:
            eligibility_source = "none"

        if eligibility_source == "none":
            stitched_predictions.append(base_row)
            continue

        replaced = False
        attempted: list[str] = []
        for rerun_path, rerun_map in rerun_maps:
            attempted.append(str(rerun_path))
            candidate = rerun_map.get(key)
            if candidate is None:
                continue
            candidate_class = _failure_class_for_row(
                tool_id=tool_id,
                row=candidate,
                class_map=class_map,
                use_classification_map=False,
            )
            if candidate_class == "provider_transport_runtime":
                continue

            stitched_predictions.append(candidate)
            decisions.append(
                {
                    "row_key": _key_to_dict(key),
                    "base_artifact": str(base_path),
                    "replacement_artifact": str(rerun_path),
                    "base_failure_class": base_class,
                    "replacement_failure_class": candidate_class,
                    "eligibility_source": eligibility_source,
                    "decision_rule": "first_non_provider_transport_by_rerun_order",
                    "decision_timestamp_utc": _utc_now(),
                }
            )
            replaced = True
            break

        if not replaced:
            stitched_predictions.append(base_row)
            unresolved.append(
                {
                    "row_key": _key_to_dict(key),
                    "attempted_rerun_artifacts": attempted,
                    "final_unresolved_reason": "no_non_provider_transport_candidate",
                }
            )

    stitched_doc = dict(base_doc)
    stitched_doc["predictions"] = stitched_predictions
    stitched_doc["stitch"] = {
        "stitched_at_utc": _utc_now(),
        "base": str(base_path),
        "reruns": [str(path) for path, _ in rerun_docs],
        "replacements": len(decisions),
        "unresolved": len(unresolved),
        "classification": classification_doc.get("schema_version"),
        "explicit_allowlist_filters": allowlist_filters,
    }

    audit_doc = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": base_doc.get("suite_id"),
        "tool_id": tool_id,
        "generated_at_utc": _utc_now(),
        "base_artifact": str(base_path),
        "rerun_artifacts": [str(path) for path, _ in rerun_docs],
        "explicit_allowlist_filters": allowlist_filters,
        "replacements": decisions,
        "unresolved": unresolved,
    }
    return stitched_doc, audit_doc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministically stitch partial rerun predictions.")
    ap.add_argument("--base", required=True, help="Base predictions JSON path.")
    ap.add_argument("--rerun", action="append", required=True, help="Rerun predictions JSON path; repeatable.")
    ap.add_argument("--classification", required=True, help="Failure classification JSON path.")
    ap.add_argument("--run-metadata", default=None, help="Run metadata JSON path.")
    ap.add_argument("--out", default=None, help="Output stitched predictions path.")
    ap.add_argument("--audit", default=None, help="Output stitch audit path.")
    ap.add_argument(
        "--stitch-allow-task-id",
        "--stitch-allow-task-ids",
        dest="stitch_allow_task_ids",
        action="append",
        default=None,
        help="Optional explicit stitch allowlist filter; repeat or use comma-separated values.",
    )
    ap.add_argument(
        "--stitch-allow-trial",
        "--stitch-allow-trials",
        dest="stitch_allow_trials",
        action="append",
        default=None,
        help="Optional explicit stitch allowlist filter; repeat or use comma-separated integers.",
    )
    ap.add_argument(
        "--stitch-allow-budget-tokens",
        dest="stitch_allow_budget_tokens",
        action="append",
        default=None,
        help="Optional explicit stitch allowlist filter; repeat or use comma-separated integers.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    repo_root = get_repo_root()
    base_path = Path(args.base).resolve()
    rerun_paths = [Path(p).resolve() for p in args.rerun]
    classification_path = Path(args.classification).resolve()
    run_metadata_path = Path(args.run_metadata).resolve() if args.run_metadata else None

    base_doc = _read_json_obj(base_path)
    rerun_docs = [(path, _read_json_obj(path)) for path in rerun_paths]
    classification_doc = _read_json_obj(classification_path)
    run_metadata_doc = _read_json_obj(run_metadata_path) if run_metadata_path else None
    explicit_allowlist_filters = _normalize_explicit_allowlist_filters(
        task_ids_raw=args.stitch_allow_task_ids,
        trials_raw=args.stitch_allow_trials,
        budget_tokens_raw=args.stitch_allow_budget_tokens,
    )

    stitched_doc, audit_doc = _stitch_predictions(
        base_doc=base_doc,
        rerun_docs=rerun_docs,
        classification_doc=classification_doc,
        run_metadata_doc=run_metadata_doc,
        base_path=base_path,
        explicit_allowlist_filters=explicit_allowlist_filters,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        tool_id = str(base_doc.get("tool_id") or "tool")
        out_path = bench_runs_root(repo_root) / f"h2h-{tool_id}-predictions-stitched.json"

    if args.audit:
        audit_path = Path(args.audit)
    else:
        tool_id = str(base_doc.get("tool_id") or "tool")
        audit_path = bench_runs_root(repo_root) / "stitch_audits" / f"h2h-{tool_id}-stitch-audit.json"

    write_report(out_path, stitched_doc)
    write_report(audit_path, audit_doc)
    print(out_path)
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
