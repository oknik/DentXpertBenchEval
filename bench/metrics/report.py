from typing import Any

from bench.protocol import A11_FIELD_KEYS


# A6.3 intentionally stays on the pending-review path. The other tasks have an
# approved rubric and can report complete, partial, or incomplete totals.
TOTALLED_CLINICAL_TASKS = {"A5.1", "A5.2", "A5.3", "A6.1", "A6.2", "A7.1"}
F1_CLINICAL_TASKS = {"A5.1", "A5.2", "A6.2"}


def summarize_fields(rows: list[dict[str, Any]], field_keys: tuple[str, ...]) -> dict[str, Any]:
    if any("scoring_status" in row for row in rows):
        return summarize_clinical(rows)
    total = len(rows)
    valid = sum(1 for row in rows if row.get("model_answer") is not None)
    total_score = sum(float(row.get("case_score", 0.0)) for row in rows)
    total_points = sum(float(row.get("case_points", 1.0)) for row in rows)
    case_scores = [
        {
            "id": row.get("id"),
            "score": float(row.get("case_score", 0.0)),
            "max_score": float(row.get("case_points", 100.0)),
        }
        for row in rows
    ]

    by_field = {}
    for key in field_keys:
        correct = sum(1 for row in rows if row.get("field_correct", {}).get(key))
        by_field[key] = {
            "correct": correct,
            "total": total,
            "score": sum(float(row.get("field_scores", {}).get(key, 0.0)) for row in rows),
            "accuracy": correct / total if total else 0.0,
        }

    total_fields = total * len(field_keys)
    correct_fields = sum(item["correct"] for item in by_field.values())
    summary = {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "case_scores": case_scores,
        "field_accuracy": correct_fields / total_fields if total_fields else 0.0,
        "total_score": total_score,
        "total_points": total_points,
        "score_rate": total_score / total_points if total_points else 0.0,
        "by_field": by_field,
        "average_score": total_score / total if total else 0.0,
    }
    fdi_metrics = [row.get("tooth_status_metrics", {}) for row in rows]
    evaluated_fdi_count = sum(int(item.get("evaluated_fdi_count", 0)) for item in fdi_metrics)
    correct_fdi_count = sum(int(item.get("correct_fdi_count", 0)) for item in fdi_metrics)
    if evaluated_fdi_count:
        summary["fdi_accuracy"] = correct_fdi_count / evaluated_fdi_count
        summary["fdi_correct"] = correct_fdi_count
        summary["fdi_total"] = evaluated_fdi_count
    return summary


def summarize_a11(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_fields(rows, A11_FIELD_KEYS)


def summarize_clinical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    skipped_rows = [row for row in rows if row.get("scoring_status") == "skipped"]
    rows = [row for row in rows if row.get("scoring_status") != "skipped"]
    total = len(rows)
    valid = sum(row.get("model_answer") is not None for row in rows)
    keys = sorted({key for row in rows for key in row.get("objective_metrics", {})})
    objective = {}
    for key in keys:
        metrics = [row["objective_metrics"][key] for row in rows if key in row.get("objective_metrics", {})]
        correct = sum(item["exact_match"] for item in metrics)
        objective[key] = {"correct": correct, "total": len(metrics), "accuracy": correct / len(metrics)}
        if key == "images":
            per_image_matched = sum(item.get("per_image_matched", 0) for item in metrics)
            per_image_total = sum(item.get("per_image_total", 0) for item in metrics)
            objective[key].update({
                "per_image_matched": per_image_matched,
                "per_image_total": per_image_total,
                "per_image_accuracy": per_image_matched / per_image_total if per_image_total else 0.0,
            })
        if key == "diagnoses":
            tp = sum(item["true_positive"] for item in metrics)
            fp = sum(item["false_positive"] for item in metrics)
            fn = sum(item["false_negative"] for item in metrics)
            objective[key].update({
                "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
                "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
                "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            })
    result = {
        "total_cases_including_skipped": total + len(skipped_rows),
        "skipped": len(skipped_rows),
        "skipped_cases": [{"id": row.get("id"), "reason": row.get("skip_reason")} for row in skipped_rows],
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "objective_metrics": objective,
        "pending_review_cases": sum(bool(row.get("pending_review_fields")) for row in rows),
        "pending_review_fields": sorted({key for row in rows for key in row.get("pending_review_fields", [])}),
        "scoring_status": "pending_review" if any(row.get("pending_review_fields") for row in rows) else "objective_only",
        "total_score": None,
        "total_points": None,
        "score_rate": None,
        "average_score": None,
        "scoring_note": "No final clinical score: review open answers and approve task scoring weights first.",
    }
    if any(row.get("task_code") in TOTALLED_CLINICAL_TASKS for row in rows):
        scored = [row for row in rows if row.get("scoring_status") == "scored"]
        result["fully_scored"] = len(scored)
        result["unscored"] = total - len(scored)
        # Invalid answers are counted as unscored, not as a medical score of 0.
        result["invalid_unscored"] = sum(
            1 for row in rows if row.get("scoring_status") == "invalid"
        )
        f1_values = [row.get("objective_metrics", {}).get("diagnoses", {}).get("f1") for row in rows]
        f1_values = [value for value in f1_values if value is not None]
        if any(row.get("task_code") in F1_CLINICAL_TASKS for row in rows):
            result["diagnosis_f1_mean"] = sum(f1_values) / len(f1_values) if f1_values else None
        if scored:
            result["scored_total_points"] = sum(row["case_points"] for row in scored)
            result["scored_total_score"] = sum(row["case_score"] for row in scored)
            result["scored_average_score"] = result["scored_total_score"] / len(scored)
        if scored and len(scored) == total:
            result["scoring_status"] = "scored"
            result["total_score"] = result["scored_total_score"]
            result["total_points"] = result["scored_total_points"]
            result["score_rate"] = result["total_score"] / result["total_points"] if result["total_points"] else None
            result["average_score"] = result["scored_average_score"]
        else:
            result["scoring_status"] = "incomplete"
        result["scoring_note"] = "Totals exclude skipped cases and remain null until all attempted cases are scored."
    return result

