import argparse
import json
from pathlib import Path
from typing import Any

from bench.metrics.report import summarize_fields
from run import TASK_SPECS, evaluate_case
from bench.protocol import TASK_FIELD_KEYS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a saved DentXpertBench model-answer JSONL file.")
    parser.add_argument("path", type=Path, help="Answer file produced by run.py")
    parser.add_argument("--task", choices=TASK_FIELD_KEYS.keys(), default=None)
    parser.add_argument("--cases", type=Path, default=None, help="Override the task case JSON path.")
    parser.add_argument("--output", type=Path, default=None, help="Scored output JSONL. Defaults to <answer>-eval.jsonl.")
    args = parser.parse_args()

    rows = load_jsonl(args.path)
    task_code = args.task or (rows[0].get("task_code") if rows else None)
    if task_code not in TASK_FIELD_KEYS:
        raise SystemExit(f"Cannot infer supported task from {args.path}. Pass --task.")

    spec = dict(TASK_SPECS[task_code])
    case_path = args.cases or Path(spec["case_path"])
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    cases = {str(case.get("case_code")): case for case in payload["cases"] if case.get("case_code") is not None}
    cases.update({str(case.get("id")): case for case in payload["cases"] if case.get("id") is not None})

    scored_rows = []
    missing = []
    for row in rows:
        case_code = row.get("case_code")
        lookup_key = str(case_code if case_code is not None else row.get("id"))
        case = cases.get(lookup_key)
        if case is None:
            missing.append(lookup_key)
            continue
        scored = evaluate_case(
            case,
            row.get("raw_response", row.get("model_response")),
            spec["answer_parser"],
            run_metadata=row.get("run_metadata"),
            token_usage=row.get("token_usage"),
        )
        # Keep the complete answer artifact alongside its calculated score.
        for key in ("thinking", "parsed_answer", "request_parameters", "raw_response"):
            if key in row:
                scored[key] = row[key]
        scored_rows.append(scored)

    if missing:
        raise SystemExit(f"Case ids not found in {case_path}: {', '.join(missing[:10])}")

    output = args.output or args.path.with_name(f"{args.path.stem}-eval{args.path.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_fields(scored_rows, TASK_FIELD_KEYS[task_code])
    summary["task_code"] = task_code
    summary["source_answers"] = str(args.path)
    summary["output"] = str(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
