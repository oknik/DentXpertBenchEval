import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.media import image_to_data_url
from bench.models import chat_completion
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
    A41_FIELD_KEYS,
    A42_FIELD_KEYS,
    A43_FIELD_KEYS,
    build_a11_prompt,
    build_a12_prompt,
    build_a13_prompt,
    build_a21_prompt,
    build_a22_prompt,
    build_a23_prompt,
    build_a24_prompt,
    build_a31_prompt,
    build_a32_prompt,
    build_a33_prompt,
    build_a34_prompt,
    select_a24_cephalometric_structures,
    select_a24_periapical_target_fdis,
    fields_to_dict,
    parse_a11_answer,
    parse_a12_answer,
    parse_a13_answer,
    parse_a21_answer,
    parse_a22_answer,
    parse_a23_answer,
    parse_a24_answer,
    parse_a31_answer,
    parse_a32_answer,
    parse_a33_answer,
    parse_a34_answer,
    build_a41_prompt,
    parse_a41_answer,
    build_a42_prompt,
    parse_a42_answer,
    build_a43_prompt,
    parse_a43_answer,
    CLINICAL_FIELD_KEYS,
    build_clinical_prompt,
    parse_clinical_answer,
    clinical_media,
    clinical_reference_answer,
    validate_clinical_answer,
    build_clinical_mock_response,
)
from bench.overlay import draw_bbox_on_image, get_case_bbox
from bench.metrics.classification import (
    score_a12_fields,
    score_a13_fields,
    score_a11_fields,
    score_a43_fields,
    score_a21_fields,
    score_a23_fields,
    score_fields,
)
from bench.metrics.localization import score_a22_fields, score_a24_fields, score_a31_fields, score_a32_fields, score_a33_fields, score_a34_fields
from bench.metrics.report import summarize_fields
from bench.metrics.clinical import score_clinical_fields
from bench.metrics.a51 import score_a51_fields, build_reason_judge_messages, parse_reason_judgment, apply_reason_judgment
from bench.metrics.a52_a53 import score_a52_fields, score_a53_fields, build_a52_judge_messages, parse_a52_judgment, apply_clinical_judgment
from bench.metrics.a61_a62_a71 import (
    apply_a71_judgment,
    build_a62_judge_messages,
    build_a71_judge_messages,
    score_a61_fields,
    score_a62_fields,
    score_a71_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_IMAGE_ROOT = REPO_ROOT / "公开数据集/公开数据集"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model-deepseek-flash.json"
TASK_SPECS = {
    "A1.1": {
        "case_path": PROJECT_ROOT / "tasks/A1.1/DentXpertBench-cases-v3.2-A1.1-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a1_1_eval_results.jsonl",
        "field_keys": A11_FIELD_KEYS,
        "prompt_builder": build_a11_prompt,
        "answer_parser": parse_a11_answer,
        "scorer": score_a11_fields,
    },
    "A1.2": {
        "case_path": PROJECT_ROOT / "tasks/A1.2/DentXpertBench-cases-v3.2-A1.2-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a1_2_eval_results.jsonl",
        "field_keys": A12_FIELD_KEYS,
        "prompt_builder": build_a12_prompt,
        "answer_parser": parse_a12_answer,
        "scorer": score_a12_fields,
    },
    "A1.3": {
        "case_path": PROJECT_ROOT / "tasks/A1.3/DentXpertBench-cases-v3.2-A1.3-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a1_3_eval_results.jsonl",
        "field_keys": A13_FIELD_KEYS,
        "prompt_builder": build_a13_prompt,
        "answer_parser": parse_a13_answer,
        "scorer": score_a13_fields,
    },
    "A2.1": {
        "case_path": PROJECT_ROOT / "tasks/A2.1/DentXpertBench-cases-v3.2-A2.1-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a2_1_eval_results.jsonl",
        "field_keys": A21_FIELD_KEYS,
        "prompt_builder": build_a21_prompt,
        "answer_parser": parse_a21_answer,
        "scorer": score_a21_fields,
    },
    "A2.2": {
        "case_path": PROJECT_ROOT / "tasks/A2.2/DentXpertBench-cases-v3.2-A2.2-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a2_2_eval_results.jsonl",
        "field_keys": A22_FIELD_KEYS,
        "prompt_builder": build_a22_prompt,
        "answer_parser": parse_a22_answer,
        "scorer": score_a22_fields,
    },
    "A2.3": {
        "case_path": PROJECT_ROOT / "tasks/A2.3/DentXpertBench-cases-v3.2-A2.3-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a2_3_eval_results.jsonl",
        "field_keys": A23_FIELD_KEYS,
        "prompt_builder": build_a23_prompt,
        "answer_parser": parse_a23_answer,
        "scorer": score_a23_fields,
    },
    "A2.4": {
        "case_path": PROJECT_ROOT / "tasks/A2.4/DentXpertBench-cases-v3.2-A2.4-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a2_4_eval_results.jsonl",
        "field_keys": A24_FIELD_KEYS,
        "prompt_builder": build_a24_prompt,
        "answer_parser": parse_a24_answer,
        "scorer": score_a24_fields,
    },
    "A3.1": {
        "case_path": PROJECT_ROOT / "tasks/A3.1/DentXpertBench-cases-v3.2-A3.1-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a3_1_eval_results.jsonl",
        "field_keys": A31_FIELD_KEYS,
        "prompt_builder": build_a31_prompt,
        "answer_parser": parse_a31_answer,
        "scorer": score_a31_fields,
    },
    "A3.2": {
        "case_path": PROJECT_ROOT / "tasks/A3.2/DentXpertBench-cases-v3.2-A3.2-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a3_2_eval_results.jsonl",
        "field_keys": A32_FIELD_KEYS,
        "prompt_builder": build_a32_prompt,
        "answer_parser": parse_a32_answer,
        "scorer": score_a32_fields,
    },
    "A3.3": {
        "case_path": PROJECT_ROOT / "tasks/A3.3/DentXpertBench-cases-v3.2-A3.3-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a3_3_eval_results.jsonl",
        "field_keys": A33_FIELD_KEYS,
        "prompt_builder": build_a33_prompt,
        "answer_parser": parse_a33_answer,
        "scorer": score_a33_fields,
    },
    "A3.4": {
        "case_path": PROJECT_ROOT / "tasks/A3.4/DentXpertBench-cases-v3.2-A3.4-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a3_4_eval_results.jsonl",
        "field_keys": A34_FIELD_KEYS,
        "prompt_builder": build_a34_prompt,
        "answer_parser": parse_a34_answer,
        "scorer": score_a34_fields,
    },
    "A4.1": {
        "case_path": PROJECT_ROOT / "tasks/A4.1/DentXpertBench-cases-v3.2-A4.1-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a4_1_eval_results.jsonl",
        "field_keys": A41_FIELD_KEYS,
        "prompt_builder": build_a41_prompt,
        "answer_parser": parse_a41_answer,
        "scorer": score_fields,
    },
    "A4.2": {
        "case_path": PROJECT_ROOT / "tasks/A4.2/DentXpertBench-cases-v3.2-A4.2-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a4_2_eval_results.jsonl",
        "field_keys": A42_FIELD_KEYS,
        "prompt_builder": build_a42_prompt,
        "answer_parser": parse_a42_answer,
        "scorer": score_fields,
    },
    "A4.3": {
        "case_path": PROJECT_ROOT / "tasks/A4.3/DentXpertBench-cases-v3.2-A4.3-2026-10-29.json",
        "output_path": PROJECT_ROOT / "results/a4_3_eval_results.jsonl",
        "field_keys": A43_FIELD_KEYS,
        "prompt_builder": build_a43_prompt,
        "answer_parser": parse_a43_answer,
        "scorer": score_a43_fields,
    },
}

for _code, _keys in CLINICAL_FIELD_KEYS.items():
    TASK_SPECS[_code] = {
        # The supplied A5-A7 case bundles are all dated 2026-10-25; use the
        # actual repository filenames instead of a stale date placeholder.
        "case_path": PROJECT_ROOT / f"tasks/{_code}/DentXpertBench-cases-v3.2-{_code}-2026-10-25.json",
        "output_path": PROJECT_ROOT / f"results/{_code.lower().replace('.', '_')}_eval_results.jsonl",
        "field_keys": _keys,
        "prompt_builder": build_clinical_prompt,
        "answer_parser": partial(parse_clinical_answer, task_code=_code),
        "scorer": partial(score_clinical_fields, task_code=_code),
    }
TASK_SPECS["A5.1"]["scorer"] = score_a51_fields
TASK_SPECS["A5.2"]["scorer"] = score_a52_fields
TASK_SPECS["A5.3"]["scorer"] = score_a53_fields
TASK_SPECS["A6.1"]["scorer"] = score_a61_fields
TASK_SPECS["A6.2"]["scorer"] = score_a62_fields
TASK_SPECS["A7.1"]["scorer"] = score_a71_fields

# A6.1 is fully objective and must never call a judge. A6.3 remains pending
# review and is unchanged by this integration.
JUDGE_TASKS = {"A5.1", "A5.2", "A5.3", "A6.2", "A7.1"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_model_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Missing model config: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid model config: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Model config must be a JSON object: {config_path}")
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise SystemExit("Model config field 'generation' must be an object")
    for key in ("model", "base_url"):
        if not str(config.get(key, "")).strip():
            raise SystemExit(f"Model config field '{key}' is required")
    return config


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def resolve_max_tokens(args: argparse.Namespace, task_code: str | None = None) -> int | None:
    if args.max_tokens is not None:
        return args.max_tokens
    return None


def build_openai_messages(
    case: dict[str, Any],
    image_root: Path,
    prompt_builder: Any,
    debug: bool = False,
) -> list[dict[str, Any]]:
    '''
    从 case 提取图片路径和任务信息，构造 prompt，返回图片和文本组成的 messages 列表
    '''
    debug_log(debug, "call build_openai_messages(case, image_root)")
    code = case.get("task_code")
    images = case["input"].get("images", [])
    if code == "A7.1" and images:
        raise ValueError("A7.1 only supports text patient records")
    debug_log(debug, f"call {getattr(prompt_builder, '__name__', 'prompt_builder')}(case)")
    prompt = prompt_builder(case)
    if debug:
        print("[debug] prompt begin")
        print(prompt)
        print("[debug] prompt end")

    media = clinical_media(case) if code in CLINICAL_FIELD_KEYS else []
    if not images and code != "A7.1":
        raise ValueError(f"task {code} requires input images")
    content = [{"type": "text", "text": prompt}]
    for index, image_meta in enumerate(images):
        image_path = image_root / image_meta["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        model_image_path = image_path
        bbox = None
        if code in {"A4.2", "A4.3"}:
            bbox = get_case_bbox(case)
        elif code in {"A5.1", "A5.2"}:
            bbox = media[index].get("diagnosis_bbox")
            if bbox is None:
                raise ValueError(f"missing diagnosis_bbox for {media[index]['evidence_id']}")
        if bbox is not None:
            safe_id = re.sub(r'[^\w.-]', '_', str(case['id']))
            overlay_path = PROJECT_ROOT / "preview" / f"{code.lower()}_model_input" / f"{safe_id}_{index + 1}.jpg"
            model_image_path = draw_bbox_on_image(image_path, bbox, overlay_path)
        if media:
            content.append({"type": "text", "text": f"影像 {media[index]['evidence_id']}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(model_image_path)}})

    return [
        {
            "role": "system",
            "content": "你是严谨的口腔多模态评测模型，只输出符合要求的最终答案。",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def call_with_retry(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    task_code: str | None = None,
) -> dict[str, Any] | None:
    '''
    调用 chat_completion()，也就是调用 api
    '''
    max_tokens = resolve_max_tokens(args, task_code)

    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            return chat_completion(
                messages=messages,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=max_tokens,
                timeout=args.timeout,
                extra_body=args.extra_body,
            )
        except urllib.error.HTTPError as exc:
            # ChatAnywhere usually returns a JSON body explaining 401/403/429,
            # but urllib's default string only contains the status line.
            try:
                error_body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                error_body = ""
            detail = f"HTTP {exc.code} {exc.reason}"
            if error_body:
                detail += f": {error_body}"
            last_error = RuntimeError(detail)
            # Keep the batch alive even for non-retryable HTTP errors. The case
            # is recorded as invalid after the configured number of attempts.
            if attempt >= args.retries:
                break
            delay = args.retry_sleep * (attempt + 1)
            print(f"[retry] {task_code or 'task'} attempt {attempt + 1}/{args.retries + 1}: {detail}; sleeping {delay:.1f}s")
            time.sleep(delay)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_sleep * (attempt + 1)
            print(f"[retry] {task_code or 'task'} attempt {attempt + 1}/{args.retries + 1}: {exc}; sleeping {delay:.1f}s")
            time.sleep(delay)
        except Exception as exc:
            # A single malformed/unsupported provider response must not abort
            # the whole benchmark. Retry it, then treat this case as invalid.
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_sleep * (attempt + 1)
            print(f"[retry] {task_code or 'task'} attempt {attempt + 1}/{args.retries + 1}: {exc}; sleeping {delay:.1f}s")
            time.sleep(delay)
    print(f"[invalid] {task_code or 'task'} model call failed after {args.retries + 1} attempts: {last_error}")
    return None


def evaluate_case(
    case: dict[str, Any],
    response: str | None,
    answer_parser: Any,
    debug: bool = False,
    run_metadata: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug_log(debug, "call evaluate_case(case, response)")
    debug_log(debug, "call fields_to_dict(reference_answer.fields)")
    task_code = case.get("task_code")
    ref = clinical_reference_answer(case) if task_code in CLINICAL_FIELD_KEYS else fields_to_dict(case["reference_answer"]["fields"], task_code)
    if task_code == "A2.4":
        modality = case.get("input", {}).get("images", [{}])[0].get("modality")
        if modality == "periapical_radiograph":
            target_fdis = set(select_a24_periapical_target_fdis(case))
            ref["structures"] = [item for item in ref["structures"] if item.get("fdi") in target_fdis]
            # Periapical answers now contain one target tooth per case, so FDI
            # is supplied by the prompt and is no longer part of each item key.
            ref["structures"] = [
                {key: value for key, value in item.items() if key != "fdi"}
                for item in ref["structures"]
            ]
        elif modality == "cephalometric_radiograph":
            target_structures = set(select_a24_cephalometric_structures(case))
            if target_structures:
                ref["structures"] = [
                    item for item in ref["structures"] if item.get("structure_code") in target_structures
                ]
    debug_log(debug, f"reference_answer={json.dumps(ref, ensure_ascii=False)}")
    try:
        debug_log(debug, f"call {getattr(answer_parser, '__name__', 'answer_parser')}(response)")
        pred = answer_parser(response)
        if task_code == "A2.4" and modality == "periapical_radiograph" and pred:
            pred["structures"] = [
                {key: value for key, value in item.items() if key != "fdi"}
                for item in pred.get("structures", [])
            ]
        if task_code in CLINICAL_FIELD_KEYS:
            validate_clinical_answer(pred, case)
        debug_log(debug, f"parsed model_answer={json.dumps(pred, ensure_ascii=False)}")
        parse_error = None
    except Exception as exc:
        pred = None
        parse_error = str(exc)
        debug_log(debug, f"parse error={parse_error}")
        response_text = response or ""
        debug_log(debug, f"raw response length={len(response_text)}")
        if response_text:
            debug_log(debug, f"raw response preview={response_text[:500]}")

    debug_log(debug, "call score_fields(model_answer, reference_answer)")
    field_keys = TASK_SPECS.get(task_code, {}).get("field_keys")
    scorer = TASK_SPECS.get(task_code, {}).get("scorer", score_fields)
    scoring = resolve_scoring(case)
    case_points = float(scoring.get("case_points", 1.0))
    if scorer is score_fields:
        score = scorer(pred, ref, case_points=case_points, field_keys=field_keys, scoring=scoring)
    else:
        score = scorer(pred, ref, case_points=case_points, scoring=scoring)
    debug_log(debug, f"score={json.dumps(score, ensure_ascii=False)}")
    return {
        "id": case.get("id"),
        "task_code": case.get("task_code"),
        "dataset_id": case.get("dataset_id"),
        "difficulty": case.get("difficulty"),
        "image_path": next((item.get("path") for item in case["input"].get("images", [])), None),
        "image_paths": [item["path"] for item in case["input"].get("images", [])],
        "reference_answer": ref,
        "reference_answer_native": case["reference_answer"],
        "model_response": response,
        "token_usage": token_usage or {},
        "model_answer": pred,
        "parse_error": parse_error,
        "run_metadata": run_metadata or {},
        **score,
    }


def resolve_scoring(case: dict[str, Any]) -> dict[str, Any]:
    """Load the authoritative scoring block from tasks/<task>/<task>.json."""
    task_code = case.get("task_code")
    if not task_code:
        return {}
    definition_path = PROJECT_ROOT / "tasks" / task_code / f"{task_code}.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    scoring = definition.get("scoring", {})

    if task_code == "A2.2":
        mode = next(
            (
                field.get("value")
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "localization_mode"
            ),
            "inventory",
        )
        return definition.get("modes", {}).get(mode, {}).get("scoring", scoring)

    if task_code == "A2.4":
        modality = case.get("input", {}).get("images", [{}])[0].get("modality")
        mode_scoring = definition.get("modes", {}).get(modality, {}).get("scoring", {})
        return {**scoring, **mode_scoring}

    if task_code == "A3.1":
        mode = next(
            (
                field.get("value", {}).get("localization_mode")
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            "local",
        )
        aliases = {"site_only": "local", "fdi": "local", "tooth": "local", "local": "local", "bbox": "bbox", "box": "bbox", "region": "region"}
        mode_scoring = definition.get("modes", {}).get(aliases.get(str(mode), str(mode)), {}).get("scoring", {})
        return {**scoring, **mode_scoring}

    if task_code == "A3.2":
        mode = next(
            (
                field.get("value", {}).get("localization_mode")
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            "region",
        )
        aliases = {"site_only": "region", "region": "region", "bbox": "bbox", "box": "bbox"}
        mode_scoring = definition.get("modes", {}).get(aliases.get(str(mode), str(mode)), {}).get("scoring", {})
        return {**scoring, **mode_scoring}

    if task_code == "A3.3":
        mode = next(
            (
                field.get("value", {}).get("localization_mode")
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            None,
        )
        scope = next(
            (
                field.get("value", {})
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            {},
        )
        if not mode:
            mode = "tooth" if scope.get("site", {}).get("type") == "tooth" else "region"
        aliases = {"site_only": "region", "region": "region", "tooth": "tooth", "fdi": "tooth", "bbox": "bbox", "box": "bbox"}
        mode_scoring = definition.get("modes", {}).get(aliases.get(str(mode), str(mode)), {}).get("scoring", {})
        return {**scoring, **mode_scoring}

    if task_code == "A3.4":
        mode = next(
            (
                field.get("value", {}).get("localization_mode")
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            None,
        )
        scope = next(
            (
                field.get("value", {})
                for field in case.get("input", {}).get("fields", [])
                if field.get("key") == "evaluation_scope" and isinstance(field.get("value"), dict)
            ),
            {},
        )
        if not mode:
            mode = "tooth" if scope.get("site", {}).get("type") == "tooth" else "region"
        aliases = {"site_only": "region", "region": "region", "tooth": "tooth", "fdi": "tooth", "bbox": "bbox", "box": "bbox"}
        mode_scoring = definition.get("modes", {}).get(aliases.get(str(mode), str(mode)), {}).get("scoring", {})
        return {**scoring, **mode_scoring}

    return scoring


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mock_response(case: dict[str, Any]) -> str:
    if case.get("task_code") in CLINICAL_FIELD_KEYS:
        return build_clinical_mock_response(case)
    ref = fields_to_dict(case["reference_answer"]["fields"], case.get("task_code"))
    task_code = case.get("task_code")
    if task_code == "A1.1":
        response = {
            "modality_type": ref["modality_type"],
            "field_of_view": ref["field_of_view"],
        }
    elif task_code == "A1.2":
        response = {
            "coverage_codes": ref["coverage_codes"],
            "quality_grade": ref["quality_grade"],
        }
    elif task_code == "A1.3":
        response = {
            "sufficiency": ref["sufficiency"],
            "minimum_necessary_action": {
                "action_code": ref["minimum_necessary_action"],
            },
        }
    elif task_code == "A2.1":
        response = {
            "dentition_stage": ref["dentition_stage"],
        }
    elif task_code == "A2.2":
        teeth = ref["teeth"]
        if teeth and isinstance(teeth[0], dict):
            response = {
                "teeth": [
                    {
                        "fdi": item["fdi"],
                        "bbox": item["bbox"],
                    }
                    for item in teeth
                ],
            }
        else:
            response = {"teeth": teeth}
    elif task_code == "A2.3":
        response = {
            "teeth": [
                {
                    "fdi": item["fdi"],
                    "status": item["status"],
                }
                for item in ref["teeth"]
            ]
        }
    elif task_code == "A2.4":
        modality = case.get("input", {}).get("images", [{}])[0].get("modality")
        if modality == "periapical_radiograph":
            target_fdis = set(select_a24_periapical_target_fdis(case))
            structures = [item for item in ref["structures"] if item.get("fdi") in target_fdis]
            structures = [
                {key: value for key, value in item.items() if key != "fdi"}
                for item in structures
            ]
        elif modality == "cephalometric_radiograph":
            target_structures = set(select_a24_cephalometric_structures(case))
            structures = [
                item for item in ref["structures"]
                if not target_structures or item.get("structure_code") in target_structures
            ]
        else:
            structures = ref["structures"]
        response = {"structures": structures}
    elif task_code == "A3.1":
        response = {"findings": ref["findings"]}
    elif task_code == "A3.2":
        response = {"findings": ref.get("findings", [])}
    elif task_code == "A3.3":
        response = {"findings": ref.get("findings", [])}
    elif task_code == "A3.4":
        response = {"findings": ref.get("findings", [])}
    elif task_code == "A4.1":
        response = {"grade_value": ref["grade_value"]}
    elif task_code == "A4.2":
        response = {"grade_value": ref["grade_value"]}
    elif task_code == "A4.3":
        response = {"surface_grades": ref["surface_grades"]}
    else:
        raise ValueError(f"unsupported task_code for mock response: {task_code}")
    return json.dumps(response, ensure_ascii=False)


def read_api_key_file(path: Path) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8-sig").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            line = value.strip()
        return line.strip().strip('"').strip("'") or None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DentXpertBench tasks with an OpenAI-compatible vision model.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="Model/API profile JSON to use. Defaults to configs/model.json.",
    )
    parser.add_argument("--task", choices=[*TASK_SPECS.keys(), "all"], default="A1.1")
    parser.add_argument("--cases", type=Path, default=None, help="Override the default case JSON path. Only valid for a single task.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--judge-model-config", type=Path, default=PROJECT_ROOT / "configs/model-gpt-6-luna.json", help="A5.1-A7.1 independent reason/plan judge config (independent of evaluated model).")
    parser.add_argument("--output", type=Path, default=None, help="Override the default output JSONL path. Only valid for a single task.")
    parser.add_argument(
        "--include-model-in-output",
        action="store_true",
        help="Include a sanitized model name in automatically generated result filenames.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override the model output token budget.")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-sleep", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent cases per task. Defaults to 32.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and check images without calling the model.")
    parser.add_argument("--debug", action="store_true", help="Print function-level dry-run trace, prompt text, mock response, parsing, and scoring.")
    parser.add_argument("--mock-response", default=None, help="Mock LLM JSON response used in dry-run. Defaults to a correct answer generated from the reference.")
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    '''
    如果 task=all 就不能使用 --cases 和 --output 参数，因为这些参数只适用于单个任务
    '''
    if args.task == "all":
        if args.cases or args.output:
            raise SystemExit("--cases and --output can only be used with a single --task.")
        return [(task_code, TASK_SPECS[task_code]) for task_code in TASK_SPECS]

    spec = dict(TASK_SPECS[args.task])
    if args.cases:
        spec["case_path"] = args.cases
    if args.output:
        spec["output_path"] = args.output
    return [(args.task, spec)]


def run_task(task_code: str, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    '''
    执行单个任务
    '''
    case_path = Path(spec["case_path"])
    output_path = Path(spec["output_path"])
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    cases = payload["cases"][: args.limit]
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    workers = 1 if args.dry_run else min(args.workers, max(1, len(cases)))
    judge_config = None
    judge_key = None
    judge_setup_error = None
    if task_code in JUDGE_TASKS and not args.dry_run:
        try:
            judge_config = load_model_config(args.judge_model_config)
            key_env = str(judge_config.get("api_key_env", "SUBLYX_API_KEY"))
            judge_key = os.getenv(key_env)
            if not judge_key and judge_config.get("api_key_file"):
                judge_key = read_api_key_file(Path(str(judge_config["api_key_file"])))
            if not judge_key and judge_config.get("requires_api_key", True):
                judge_setup_error = f"Missing judge API key; configure {key_env} or judge key file"
        except (SystemExit, ValueError, OSError) as exc:
            judge_setup_error = str(exc)
    run_metadata = {
        "run_id": args.run_id,
        "started_at": args.run_started_at,
        "task_code": task_code,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": resolve_max_tokens(args, task_code),
        "workers": workers,
    }

    def process_case(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, case = item
        if task_code == "A5.1" and case.get("id") == "candidate:e1cf8e01197540aea86895416ae5f8a7" and any(not m.get("diagnosis_bbox") for m in clinical_media(case)):
            print(f"[skipped] {task_code} {case['id']}: diagnosis_bbox missing")
            return index, {"id": case["id"], "task_code": task_code, "model_answer": None, "case_score": None, "scoring_status": "skipped", "skip_reason": "diagnosis_bbox missing; awaiting annotation", "pending_review_fields": [], "objective_metrics": {}, "run_metadata": run_metadata}
        try:
            token_usage = None
            debug_log(args.debug, f"{task_code} case {index}/{len(cases)} id={case['id']}")
            messages = build_openai_messages(case, args.image_root, spec["prompt_builder"], debug=args.debug)
            if args.dry_run:
                response = args.mock_response if args.mock_response is not None else build_mock_response(case)
                print(f"[dry-run] {task_code} {index}/{len(cases)} {case['id']} image ok")
                if args.debug:
                    print("[debug] would call call_with_retry(messages, args), skipped because --dry-run is set")
                    print(f"[debug] resolved max_tokens={resolve_max_tokens(args, task_code)}")
                    print(f"[debug] mock LLM response={response}")
            else:
                print(f"[eval] {task_code} {index}/{len(cases)} {case['id']}")
                debug_log(args.debug, f"resolved max_tokens={resolve_max_tokens(args, task_code)}")
                call_result = call_with_retry(messages, args, task_code=task_code)
                response = call_result.get("content", "") if call_result else None
                token_usage = call_result.get("usage") if call_result else None
                if token_usage:
                    print(
                        f"[usage] {task_code} {case['id']} "
                        f"input={token_usage.get('prompt_tokens', '?')} "
                        f"thinking={token_usage.get('reasoning_tokens', '?')} "
                        f"output={token_usage.get('visible_completion_tokens', token_usage.get('completion_tokens', '?'))} "
                        f"total={token_usage.get('total_tokens', '?')}"
                    )
                elif call_result is not None:
                    print(f"[usage] {task_code} {case['id']} unavailable (provider omitted usage)")
            result = evaluate_case(
                case,
                response,
                spec["answer_parser"],
                debug=args.debug,
                run_metadata=run_metadata,
                token_usage=token_usage if not args.dry_run else None,
            )
            if task_code in JUDGE_TASKS and result.get("pending_review_fields") and result["model_answer"] is not None and not args.dry_run:
                result["reason_judge_model"] = judge_config.get("model") if judge_config else None
                result["reason_judge_error"] = judge_setup_error
                if judge_setup_error:
                    print(f"[judge-error] {case['id']}: {judge_setup_error}")
                if not judge_setup_error and judge_config:
                    try:
                        generation = judge_config.get("generation", {})
                        if task_code == "A5.2":
                            judge_messages = build_a52_judge_messages(result["reference_answer"], result["model_answer"])
                        elif task_code == "A6.2":
                            judge_messages = build_a62_judge_messages(result["reference_answer"], result["model_answer"])
                        elif task_code == "A7.1":
                            judge_messages = build_a71_judge_messages(
                                build_clinical_prompt(case),
                                result["reference_answer"]["plans"],
                                result["model_answer"]["plans"],
                            )
                        else:
                            judge_messages = build_reason_judge_messages(result["reference_answer"]["reason"], result["model_answer"]["reason"])
                        judgment_response = chat_completion(
                            messages=judge_messages,
                            model=str(judge_config["model"]), api_key=judge_key, base_url=str(judge_config["base_url"]),
                            temperature=0, max_tokens=1024, timeout=min(int(generation.get("timeout", 120)), 120),
                            extra_body=generation.get("extra_body", {}),
                        )
                        result["reason_judge_response"] = judgment_response.get("content")
                        result["reason_judge_token_usage"] = judgment_response.get("usage") or {}
                        if task_code == "A5.1":
                            apply_reason_judgment(result, parse_reason_judgment(judgment_response.get("content")))
                        elif task_code == "A5.2":
                            apply_clinical_judgment(result, parse_a52_judgment(judgment_response.get("content")))
                        elif task_code == "A6.2":
                            apply_clinical_judgment(result, parse_reason_judgment(judgment_response.get("content")))
                        elif task_code == "A7.1":
                            apply_a71_judgment(result, parse_reason_judgment(judgment_response.get("content")))
                        else:
                            apply_clinical_judgment(result, parse_reason_judgment(judgment_response.get("content")))
                    except Exception as exc:
                        result["reason_judge_error"] = str(exc)
                        print(f"[judge-error] {case['id']}: {exc}")
        except Exception as exc:
            # Never let one malformed case abort the whole task batch.
            print(f"[invalid] {task_code} {case.get('id', index)}: {exc}")
            result = evaluate_case(
                case,
                None,
                spec["answer_parser"],
                debug=args.debug,
                run_metadata=run_metadata,
                token_usage=None,
            )
            result["parse_error"] = f"case processing error: {exc}"
        return index, result

    results_by_index: dict[int, dict[str, Any]] = {}
    items = list(enumerate(cases, 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_case, item) for item in items]
        for future in as_completed(futures):
            index, result = future.result()
            results_by_index[index] = result
            if not args.dry_run:
                save_jsonl(output_path, [results_by_index[i] for i in sorted(results_by_index)])

    results = [results_by_index[i] for i in sorted(results_by_index)]

    if args.dry_run:
        invalid = [row for row in results if row.get("scoring_status") != "skipped" and row["model_answer"] is None]
        if invalid:
            raise SystemExit(f"Dry run failed for {task_code}: {len(invalid)}/{len(cases)} invalid cases. First error: {invalid[0]['parse_error']}")
        skipped = sum(row.get("scoring_status") == "skipped" for row in results)
        print(f"Dry run passed for {task_code}: {len(cases) - skipped} cases, {skipped} skipped. No result files were written.")
        return None

    summary = summarize_fields(results, spec["field_keys"])
    usage_rows = [row.get("token_usage", {}) for row in results if row.get("token_usage", {}).get("total_tokens") is not None]
    if usage_rows:
        totals = [int(row["total_tokens"]) for row in usage_rows]
        input_totals = [int(row["prompt_tokens"]) for row in usage_rows if row.get("prompt_tokens") is not None]
        output_totals = [int(row["completion_tokens"]) for row in usage_rows if row.get("completion_tokens") is not None]
        sorted_totals = sorted(totals)
        p95_index = min(len(sorted_totals) - 1, max(0, math.ceil(len(sorted_totals) * 0.95) - 1))
        completion_values = [int(row["completion_tokens"]) for row in usage_rows if row.get("completion_tokens") is not None]
        reasoning_values = [int(row["reasoning_tokens"]) for row in usage_rows if row.get("reasoning_tokens") is not None]
        visible_values = [int(row["visible_completion_tokens"]) for row in usage_rows if row.get("visible_completion_tokens") is not None]
        sorted_completion = sorted(completion_values)
        completion_p95 = None
        if sorted_completion:
            completion_p95 = sorted_completion[
                min(len(sorted_completion) - 1, max(0, math.ceil(len(sorted_completion) * 0.95) - 1))
            ]
        summary["token_usage"] = {
            "measured_cases": len(usage_rows),
            "missing_cases": len(results) - len(usage_rows),
            "total_tokens": sum(totals),
            "avg_total_tokens": round(sum(totals) / len(totals), 2),
            "max_total_tokens": max(totals),
            "p95_total_tokens": sorted_totals[p95_index],
            "max_completion_tokens": max(completion_values) if completion_values else None,
            "p95_completion_tokens": completion_p95,
            "total_prompt_tokens": sum(input_totals) if input_totals else None,
            "total_completion_tokens": sum(output_totals) if output_totals else None,
            "total_reasoning_tokens": sum(reasoning_values) if reasoning_values else None,
            "avg_reasoning_tokens": round(sum(reasoning_values) / len(reasoning_values), 2) if reasoning_values else None,
            "max_reasoning_tokens": max(reasoning_values) if reasoning_values else None,
            "total_visible_completion_tokens": sum(visible_values) if visible_values else None,
        }
    summary["run_metadata"] = run_metadata
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {task_code}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {output_path}")
    print(f"Saved summary to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    model_config = load_model_config(args.model_config)
    generation = model_config.get("generation", {})
    args.model = str(model_config["model"])
    args.base_url = str(model_config["base_url"])
    args.temperature = args.temperature if args.temperature is not None else generation.get("temperature")
    args.max_tokens = args.max_tokens if args.max_tokens is not None else generation.get("max_tokens")
    args.timeout = args.timeout if args.timeout is not None else int(generation.get("timeout", 120))
    args.retries = args.retries if args.retries is not None else int(generation.get("retries", 5))
    args.retry_sleep = args.retry_sleep if args.retry_sleep is not None else float(generation.get("retry_sleep", 2.0))
    args.extra_body = generation.get("extra_body", {})
    if not isinstance(args.extra_body, dict):
        raise SystemExit("Model config field 'generation.extra_body' must be an object")
    args.api_key_file = Path(str(model_config.get("api_key_file", "D:\\Desktop\\sublyx.txt")))
    args.api_key = os.getenv(str(model_config.get("api_key_env", "SUBLYX_API_KEY"))) or os.getenv("OPENAI_API_KEY")
    if not args.api_key and args.api_key_file:
        args.api_key = read_api_key_file(args.api_key_file)

    requires_api_key = bool(model_config.get("requires_api_key", True))
    if not args.dry_run and requires_api_key and not args.api_key:
        raise SystemExit("Missing API key. Set the configured environment variable or provide the configured key file.")

    args.run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    args.run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    tasks = resolve_tasks(args)
    if not args.output:
        output_tag = ""
        if args.include_model_in_output:
            output_tag = "_" + "".join(
                char if char.isalnum() or char in "-_" else "-"
                for char in args.model
            ).strip("-")
        for _, spec in tasks:
            default_output = Path(spec["output_path"])
            spec["output_path"] = default_output.with_name(
                f"{default_output.stem}_{args.run_id}{output_tag}{default_output.suffix}"
            )

    for task_code, spec in tasks:
        run_task(task_code, spec, args)


if __name__ == "__main__":
    main()
