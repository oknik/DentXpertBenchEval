import argparse
import json
from pathlib import Path
from typing import Any

import run
from bench.metrics.a51 import parse_reason_judgment
from bench.metrics.a52_a53 import apply_clinical_judgment
from bench.metrics.a61_a62_a71 import apply_a71_judgment
from bench.metrics.report import summarize_fields
from bench.protocol import (
    A11_FIELD_KEYS,
    A12_FIELD_KEYS,
    A13_FIELD_KEYS,
    A21_FIELD_KEYS,
    A22_FIELD_KEYS,
    A23_FIELD_KEYS,
    A24_FIELD_KEYS,
    A31_FIELD_KEYS,
    A32_FIELD_KEYS,
    A33_FIELD_KEYS,
    A34_FIELD_KEYS,
    CLINICAL_FIELD_KEYS,
)


TASK_FIELD_KEYS = {
    "A1.1": A11_FIELD_KEYS,
    "A1.2": A12_FIELD_KEYS,
    "A1.3": A13_FIELD_KEYS,
    "A2.1": A21_FIELD_KEYS,
    "A2.2": A22_FIELD_KEYS,
    "A2.3": A23_FIELD_KEYS,
    "A2.4": A24_FIELD_KEYS,
    "A3.1": A31_FIELD_KEYS,
    "A3.2": A32_FIELD_KEYS,
    "A3.3": A33_FIELD_KEYS,
    "A3.4": A34_FIELD_KEYS,
    **CLINICAL_FIELD_KEYS,
}

# A6.3 has no approved rubric yet and stays exactly as saved. The listed tasks
# can be rebuilt from stored answers; saved judgments are reused without any API
# call.
RECOMPUTE_TASKS = {"A6.1", "A6.2", "A7.1"}
COMPLETED_SCORE_FIELDS = (
    "scoring_status",
    "case_score",
    "field_scores",
    "pending_review_fields",
    "reason_judgment",
    "reason_max_points",
    "plan_max_points",
    "reason_not_applicable",
    "scoring_note",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _apply_saved_judgment(row: dict[str, Any], task_code: str) -> bool:
    judgment = row.get("reason_judgment")
    if not isinstance(judgment, dict):
        return False
    judgment = parse_reason_judgment(json.dumps(judgment, ensure_ascii=False))
    if task_code == "A6.2":
        apply_clinical_judgment(row, judgment)
    elif task_code == "A7.1":
        apply_a71_judgment(row, judgment)
    else:
        return False
    return True


def recompute_clinical_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild objective scores from stored answers and reuse saved judgments.

    No model or judge API is called. A completed saved score is preserved when
    the recomputation cannot reproduce it (for example, legacy rows that predate
    the current judgment fields), so a saved judge result is never silently lost.
    """
    task_code = row.get("task_code")
    if task_code not in RECOMPUTE_TASKS or row.get("scoring_status") == "skipped":
        return row
    spec = run.TASK_SPECS.get(task_code)
    ref = row.get("reference_answer")
    if spec is None or not isinstance(ref, dict):
        return row

    case_points = row.get("case_points")
    case_points = 100.0 if case_points is None else float(case_points)
    scoring = run.resolve_scoring({"task_code": task_code})
    recomputed = spec["scorer"](
        row.get("model_answer"),
        ref,
        case_points=case_points,
        scoring=scoring,
    )
    updated = {**row, **recomputed}

    applied = False
    if updated.get("scoring_status") == "pending_review" and isinstance(row.get("reason_judgment"), dict):
        try:
            applied = _apply_saved_judgment(updated, task_code)
        except (KeyError, TypeError, ValueError):
            applied = False

    if (
        not applied
        and updated.get("scoring_status") != "scored"
        and row.get("scoring_status") == "scored"
        and row.get("case_score") is not None
    ):
        for key in COMPLETED_SCORE_FIELDS:
            if key in row:
                updated[key] = row[key]
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute DentXpertBench metrics from saved JSONL predictions.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--task", choices=TASK_FIELD_KEYS.keys(), default=None)
    args = parser.parse_args()

    rows = [recompute_clinical_row(row) for row in load_jsonl(args.path)]
    task_code = args.task or (rows[0].get("task_code") if rows else None)
    if task_code not in TASK_FIELD_KEYS:
        raise SystemExit(f"Cannot infer supported task from {args.path}. Pass --task.")

    summary = summarize_fields(rows, TASK_FIELD_KEYS[task_code])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
