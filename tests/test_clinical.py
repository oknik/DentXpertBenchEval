import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import evaluate
import run
from bench.metrics.report import summarize_fields
from bench.protocol import (
    CLINICAL_FIELD_KEYS,
    build_clinical_mock_response,
    build_clinical_prompt,
    clinical_media,
    clinical_reference_answer,
    parse_clinical_answer,
    validate_clinical_answer,
)


def load_cases(code):
    return json.loads(run.TASK_SPECS[code]["case_path"].read_text(encoding="utf-8"))["cases"]


class ClinicalProtocolTests(unittest.TestCase):
    def test_every_case_renders_without_reference_leakage_and_evaluates(self):
        count = 0
        for code, keys in CLINICAL_FIELD_KEYS.items():
            rows = []
            for case in load_cases(code):
                with self.subTest(code=code, case=case["id"]):
                    prompt = build_clinical_prompt(case)
                    self.assertNotIn("{{", prompt)
                    changed = copy.deepcopy(case)
                    if code == "A5.2":
                        for field in changed["reference_answer"]["fields"]:
                            if field["key"] == "differential_diagnoses":
                                for entry in field["value"]:
                                    entry["role"] = "PRIVATE_LABEL"
                                    entry["reason"] = "PRIVATE_REASON"
                    else:
                        changed["reference_answer"] = {"fields": [{"key": "secret", "value": "PRIVATE_LABEL"}]}
                    self.assertEqual(prompt, build_clinical_prompt(changed))
                    row = run.evaluate_case(case, build_clinical_mock_response(case), run.TASK_SPECS[code]["answer_parser"])
                    self.assertIsNone(row["parse_error"])
                    if code == "A6.1":
                        # A6.1 is objective-only and now always scored when valid.
                        self.assertEqual(row["scoring_status"], "scored")
                        self.assertIsInstance(row["case_score"], float)
                    elif code == "A5.3" and row["scoring_status"] == "scored":
                        self.assertEqual(row["case_score"], 0)
                    else:
                        self.assertIsNone(row["case_score"])
                    self.assertEqual(row["reference_answer_native"], case["reference_answer"])
                    rows.append(row)
                    count += 1
            summary = summarize_fields(rows, keys)
            self.assertEqual(summary["valid"], len(rows))
            if code == "A6.1":
                self.assertIsNotNone(summary["score_rate"])
            else:
                self.assertIsNone(summary["score_rate"])
            self.assertIn(code, evaluate.TASK_FIELD_KEYS)
        self.assertEqual(count, 352)

    def test_all_image_messages_and_red_boxes(self):
        scratch = run.PROJECT_ROOT / "preview"
        scratch.mkdir(exist_ok=True)
        missing_boxes = []
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            for code in CLINICAL_FIELD_KEYS:
                # Exercise all annotated image metadata, including every bbox.
                for case in load_cases(code):
                    with self.subTest(code=code, case=case["id"]):
                        for image in case["input"]["images"]:
                            path = root / image["path"]
                            path.parent.mkdir(parents=True, exist_ok=True)
                            if not path.exists():
                                Image.new("RGB", (120, 120), "white").save(path)
                        if code in {"A5.1", "A5.2"} and any("diagnosis_bbox" not in m for m in clinical_media(case)):
                            with self.assertRaisesRegex(ValueError, "missing diagnosis_bbox"):
                                run.build_openai_messages(case, root, build_clinical_prompt)
                            missing_boxes.append(case["id"])
                            continue
                        with patch.object(run, "PROJECT_ROOT", root), patch.object(run, "draw_bbox_on_image", wraps=run.draw_bbox_on_image) as draw:
                            messages = run.build_openai_messages(case, root, build_clinical_prompt)
                        content = messages[1]["content"]
                        pictures = [item for item in content if item["type"] == "image_url"]
                        self.assertEqual(len(pictures), len(case["input"]["images"]))
                        labels = [item["text"] for item in content[1:] if item["type"] == "text"]
                        self.assertEqual(labels, [f"影像 {m['evidence_id']}" for m in clinical_media(case)])
                        if code in {"A5.1", "A5.2"}:
                            self.assertEqual(draw.call_count, len(pictures))
                            for call, media in zip(draw.call_args_list, clinical_media(case)):
                                self.assertEqual(call.args[1], media["diagnosis_bbox"])
                                with Image.open(call.args[2]) as image:
                                    colors = zip(*(channel.tobytes() for channel in image.split()))
                                    self.assertTrue(any(r > 160 and r > g + 50 and r > b + 50 for r, g, b in colors))
                        if code == "A7.1":
                            self.assertEqual(len(content), 1)
        self.assertEqual(missing_boxes, ["candidate:e1cf8e01197540aea86895416ae5f8a7"])

    def test_media_order_uses_paths_not_annotation_order(self):
        case = copy.deepcopy(load_cases("A6.1")[0])
        expected = [item["evidence_id"] for item in clinical_media(case)]
        for field in case["input"]["fields"]:
            if field["key"] == "evidence_media":
                field["value"].reverse()
        self.assertEqual([item["evidence_id"] for item in clinical_media(case)], expected)

    def test_unknown_diagnosis_and_missing_image_are_invalid(self):
        case = load_cases("A5.1")[0]
        response = json.dumps({"diagnoses": ["不存在的选项"], "reason": "测试"}, ensure_ascii=False)
        row = run.evaluate_case(case, response, run.TASK_SPECS["A5.1"]["answer_parser"])
        self.assertIsNone(row["model_answer"])
        self.assertIn("outside input options", row["parse_error"])
        case = load_cases("A6.1")[0]
        response = json.loads(build_clinical_mock_response(case))
        response["images"].pop()
        with self.assertRaisesRegex(ValueError, "all input M identifiers"):
            validate_clinical_answer(response, case)

    def test_strict_answer_schema_and_conditional_reason(self):
        bad = [
            ("A5.3", {"sufficiency": "充足", "reason": "多余"}),
            ("A5.3", {"sufficiency": "不足"}),
            ("A5.3", {"sufficiency": "supported"}),
            ("A6.1", {"total": True, "images": [{"image": "M01", "count": 1}]}),
            ("A6.1", {"total": -1, "images": [{"image": "M01", "count": 0}]}),
            ("A6.1", {"total": 0, "images": [{"image": "M01", "count": 1}]}),
            ("A6.3", {"conditions": [{"category": "其他", "detail": "测试"}]}),
            ("A7.1", {"plans": []}),
            ("A7.1", {"plans": [{"plan": "测试", "extra": "多余"}]}),
            ("A5.2", {"diagnoses": ["龋病"], "reason": "字" * 101, "exclude_reason": "测试"}),
        ]
        for code, obj in bad:
            with self.subTest(code=code, obj=obj), self.assertRaises(ValueError):
                parse_clinical_answer(json.dumps(obj, ensure_ascii=False), code)
        self.assertEqual(parse_clinical_answer('{"sufficiency":"充足"}', "A5.3"), {"sufficiency": "充足"})

    def test_objective_matches_and_review_status(self):
        for code in ("A5.1", "A5.2", "A6.1", "A6.2"):
            case = load_cases(code)[0]
            answer = clinical_reference_answer(case)
            if "reason" in answer:
                answer["reason"] = "相同诊断，理由用不同文字表达。"
            if "exclude_reason" in answer:
                answer["exclude_reason"] = "此项交由人工审核。"
            row = run.evaluate_case(case, json.dumps(answer, ensure_ascii=False), run.TASK_SPECS[code]["answer_parser"])
            self.assertIsNone(row["parse_error"])
            self.assertTrue(all(item["exact_match"] for item in row["objective_metrics"].values()))
            if code == "A6.1":
                self.assertEqual(row["scoring_status"], "scored")
                self.assertEqual(row["case_score"], 100)
            else:
                self.assertIsNone(row["case_score"])
        case = load_cases("A5.3")[0]
        row = run.evaluate_case(case, '{"sufficiency":"充足"}', run.TASK_SPECS["A5.3"]["answer_parser"])
        self.assertTrue(row["objective_metrics"]["sufficiency"]["exact_match"])
        self.assertEqual(row["scoring_status"], "scored")
        self.assertEqual(row["case_score"], 100)
        self.assertIn("diagnosis_support", row["reference_answer"])

    def test_text_only_cli_dry_run_and_invalid_exit_code(self):
        good = subprocess.run([sys.executable, "run.py", "--task", "A7.1", "--dry-run", "--limit", "1"], cwd=run.PROJECT_ROOT, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr.decode("utf-8"))
        bad = subprocess.run([sys.executable, "run.py", "--task", "A7.1", "--dry-run", "--limit", "1", "--mock-response", "{}"], cwd=run.PROJECT_ROOT, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn(b"Dry run failed", bad.stderr)

    def test_existing_tasks_still_build_messages_and_score(self):
        scratch = run.PROJECT_ROOT / "preview"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            for code, spec in run.TASK_SPECS.items():
                if code in CLINICAL_FIELD_KEYS:
                    continue
                with self.subTest(code=code):
                    case = json.loads(spec["case_path"].read_text(encoding="utf-8"))["cases"][0]
                    for meta in case["input"]["images"]:
                        path = root / meta["path"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (120, 120), "white").save(path)
                    with patch.object(run, "PROJECT_ROOT", root):
                        messages = run.build_openai_messages(case, root, spec["prompt_builder"])
                    self.assertTrue(messages[1]["content"])
                    result = run.evaluate_case(case, run.build_mock_response(case), spec["answer_parser"])
                    self.assertIsNone(result["parse_error"])
                    self.assertIsInstance(result["case_score"], (float, int))


if __name__ == "__main__":
    unittest.main()
