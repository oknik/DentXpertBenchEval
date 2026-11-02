"""Count text tokens and estimate vision tokens used by every case.

The tokenizer.json can count the text part of a multimodal request. Image
tokens are model/processor-specific. The default estimate follows the
common Qwen2.5-VL convention: one visual token per 28x28 image area, plus
vision_start/vision_end for each image.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from bench.protocol import build_clinical_prompt
from run import PROJECT_ROOT, TASK_SPECS


SYSTEM_PROMPT = "你是严谨的口腔多模态评测模型，只输出符合要求的最终答案。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count text tokens for all DentXpertBench input cases."
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT / "tokenizer.json",
        help="Path to a Hugging Face tokenizer.json file.",
    )
    parser.add_argument(
        "--task",
        default="all",
        help="Task code such as A1.1, or all (default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/input_token_counts.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--vision-patch-size",
        type=int,
        default=28,
        help="Patch size in pixels for image-token estimation (default: 28).",
    )
    return parser.parse_args()


def task_case_path(task_code: str) -> Path:
    if task_code in TASK_SPECS:
        configured = Path(TASK_SPECS[task_code]["case_path"])
        if configured.is_file():
            return configured
    # A7.2 is intentionally not registered in the evaluator, but its raw
    # instruction is still measurable and should not silently disappear.
    matches = sorted((PROJECT_ROOT / "tasks" / task_code).glob("DentXpertBench-cases*.json"))
    if not matches:
        raise FileNotFoundError(f"No case JSON found for {task_code}")
    return matches[0]


def prompt_for_case(task_code: str, case: dict[str, Any]) -> str:
    if task_code in TASK_SPECS:
        return str(TASK_SPECS[task_code]["prompt_builder"](case))
    if task_code == "A7.2":
        return str(case.get("input", {}).get("instruction", ""))
    raise ValueError(f"No prompt builder for {task_code}")


def image_metadata(case: dict[str, Any]) -> tuple[int, str]:
    images = case.get("input", {}).get("images", [])
    sizes = [f"{item.get('width', '?')}x{item.get('height', '?')}" for item in images]
    return len(images), ";".join(sizes)


def estimate_image_tokens(case: dict[str, Any], patch_size: int = 28) -> int:
    """Estimate visual tokens using the Qwen2.5-VL 28px merged-patch rule."""
    total = 0
    for image in case.get("input", {}).get("images", []):
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        total += ((width + patch_size - 1) // patch_size) * ((height + patch_size - 1) // patch_size)
        total += 2  # <|vision_start|> and <|vision_end|>
    return total


def p95(values: list[int]) -> int | None:
    if not values:
        return None
    return sorted(values)[max(0, int(len(values) * 0.95) - 1)]


def main() -> None:
    args = parse_args()
    if not args.tokenizer.is_file():
        raise SystemExit(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    task_codes = list(TASK_SPECS) if args.task == "all" else [args.task]
    if args.task == "all" and (PROJECT_ROOT / "tasks" / "A7.2").is_dir():
        task_codes.append("A7.2")

    rows: list[dict[str, Any]] = []
    for task_code in task_codes:
        payload = json.loads(task_case_path(task_code).read_text(encoding="utf-8"))
        for index, case in enumerate(payload.get("cases", []), 1):
            prompt = prompt_for_case(task_code, case)
            system_tokens = len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False).ids)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            image_count, image_sizes = image_metadata(case)
            estimated_image_tokens = estimate_image_tokens(case, args.vision_patch_size)
            rows.append(
                {
                    "task": task_code,
                    "index": index,
                    "id": case.get("id"),
                    "case_code": case.get("case_code"),
                    "prompt_chars": len(prompt),
                    "system_tokens": system_tokens,
                    "prompt_tokens": prompt_tokens,
                    "text_input_tokens": system_tokens + prompt_tokens,
                    "estimated_image_tokens": estimated_image_tokens,
                    "estimated_multimodal_input_tokens": system_tokens + prompt_tokens + estimated_image_tokens,
                    "image_count": image_count,
                    "image_sizes": image_sizes,
                }
            )

    summaries: dict[str, Any] = {}
    for task_code in task_codes:
        values = [row["text_input_tokens"] for row in rows if row["task"] == task_code]
        multimodal_values = [row["estimated_multimodal_input_tokens"] for row in rows if row["task"] == task_code]
        summaries[task_code] = {
            "cases": len(values),
            "min_text_input_tokens": min(values) if values else None,
            "avg_text_input_tokens": round(statistics.mean(values), 2) if values else None,
            "p95_text_input_tokens": p95(values),
            "max_text_input_tokens": max(values) if values else None,
            "total_text_input_tokens": sum(values),
            "min_estimated_multimodal_input_tokens": min(multimodal_values) if multimodal_values else None,
            "avg_estimated_multimodal_input_tokens": round(statistics.mean(multimodal_values), 2) if multimodal_values else None,
            "p95_estimated_multimodal_input_tokens": p95(multimodal_values),
            "max_estimated_multimodal_input_tokens": max(multimodal_values) if multimodal_values else None,
            "total_estimated_multimodal_input_tokens": sum(multimodal_values),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tokenizer": str(args.tokenizer),
        "image_estimator": {
            "name": "qwen2.5-vl-like-28px-merged-patch",
            "patch_size": args.vision_patch_size,
            "formula": "sum(ceil(width/patch_size)*ceil(height/patch_size)+2) per image",
            "note": "Estimate only; actual server usage depends on the deployed vision processor resize policy.",
        },
        "system_prompt": SYSTEM_PROMPT,
        "tasks": summaries,
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    all_values = [row["text_input_tokens"] for row in rows]
    all_multimodal_values = [row["estimated_multimodal_input_tokens"] for row in rows]
    print(f"Counted {len(rows)} cases across {len(task_codes)} tasks")
    print(f"Overall text input tokens: min={min(all_values)}, avg={statistics.mean(all_values):.2f}, p95={p95(all_values)}, max={max(all_values)}, total={sum(all_values)}")
    print(f"Overall estimated multimodal input: min={min(all_multimodal_values)}, avg={statistics.mean(all_multimodal_values):.2f}, p95={p95(all_multimodal_values)}, max={max(all_multimodal_values)}, total={sum(all_multimodal_values)}")
    for task_code in task_codes:
        item = summaries[task_code]
        print(f"{task_code}: text_avg={item['avg_text_input_tokens']} estimated_multimodal_avg={item['avg_estimated_multimodal_input_tokens']} estimated_multimodal_max={item['max_estimated_multimodal_input_tokens']}")
    print(f"JSON: {args.output}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
