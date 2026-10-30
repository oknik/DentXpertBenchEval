import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate
import run
from bench.metrics.a51 import parse_reason_judgment
from bench.metrics.a52_a53 import apply_clinical_judgment
from bench.metrics.a61_a62_a71 import (
    apply_a71_judgment,
    build_a62_judge_messages,
    build_a71_judge_messages,
    score_a61_fields,
    score_a62_fields,
    score_a71_fields,
)
from bench.metrics.clinical import score_clinical_fields
from bench.metrics.report import summarize_fields
from bench.protocol import (
    build_clinical_prompt,
    clinical_media,
    clinical_reference_answer,
    parse_clinical_answer,
    validate_clinical_answer,
)


def cases(code):
    return json.loads(run.TASK_SPECS[code]["case_path"].read_text(encoding="utf-8"))["cases"]


def execute_task(code, case, prediction, judge_reply=None, judge_error=None, dry_run=False):
    scratch = run.PROJECT_ROOT / "preview"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        source, output, config = root / "cases.json", root / "results.jsonl", root / "judge.json"
        source.write_text(json.dumps({"cases": [case]}, ensure_ascii=False), encoding="utf-8")
        config.write_text(
            json.dumps({"model": "test-judge", "base_url": "http://example.invalid/v1", "requires_api_key": False}),
            encoding="utf-8",
        )
        args = types.SimpleNamespace(
            limit=None, workers=1, dry_run=dry_run, judge_model_config=config,
            run_id="test", run_started_at="test", model="tested", base_url="http://example.invalid/v1",
            temperature=0, max_tokens=1000, image_root=root, debug=False, api_key=None,
            timeout=1, retries=0, retry_sleep=0, extra_body={}, mock_response=None,
        )
        spec = {**run.TASK_SPECS[code], "case_path": source, "output_path": output}
        with (
            redirect_stdout(io.StringIO()),
            patch.object(run, "build_openai_messages", return_value=[]),
            patch.object(
                run,
                "call_with_retry",
                return_value={"content": json.dumps(prediction, ensure_ascii=False)},
            ) as model_call,
            patch.object(
                run,
                "chat_completion",
                return_value={"content": judge_reply, "usage": {"total_tokens": 10}},
                side_effect=judge_error,
            ) as judge,
        ):
            summary = run.run_task(code, spec, args)
        row = None
        if output.exists():
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        return row, summary, judge.call_count, model_call.call_count, output.exists()


class A61ScoringTests(unittest.TestCase):
    def test_exact_partial_and_zero_formula(self):
        ref = {"total": 3, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 2}]}

        full = score_a61_fields(
            {"total": 3, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 2}]},
            ref,
        )
        self.assertEqual(full["case_score"], 100)
        self.assertTrue(full["objective_metrics"]["images"]["exact_match"])
        self.assertEqual(
            (full["objective_metrics"]["images"]["per_image_matched"],
             full["objective_metrics"]["images"]["per_image_total"]),
            (2, 2),
        )
        self.assertEqual(full["pending_review_fields"], [])

        total_only = score_a61_fields(
            {"total": 3, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 0}]},
            ref,
        )
        self.assertEqual(total_only["case_score"], 75)
        self.assertFalse(total_only["objective_metrics"]["images"]["exact_match"])
        self.assertEqual(total_only["objective_metrics"]["images"]["per_image_matched"], 1)

        one_image_only = score_a61_fields(
            {"total": 2, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 2}]},
            ref,
        )
        self.assertEqual(one_image_only["case_score"], 50)

        none = score_a61_fields(
            {"total": 0, "images": [{"image": "M01", "count": 0}, {"image": "M02", "count": 0}]},
            ref,
        )
        self.assertEqual(none["case_score"], 0)
        self.assertEqual(none["scoring_status"], "scored")

        invalid = score_a61_fields(None, ref)
        self.assertEqual(invalid["scoring_status"], "invalid")
        self.assertIsNone(invalid["case_score"])
        self.assertEqual(invalid["objective_metrics"], {})
        self.assertEqual(score_clinical_fields(copy.deepcopy(ref), ref, task_code="A6.1")["case_score"], 100)

    def test_zero_counts_earn_full_credit(self):
        ref = {"total": 0, "images": [{"image": "M01", "count": 0}, {"image": "M02", "count": 0}]}
        row = score_a61_fields(copy.deepcopy(ref), ref)
        self.assertEqual(row["case_score"], 100)
        self.assertEqual(row["objective_metrics"]["images"]["per_image_accuracy"], 1.0)

    def test_strict_integer_identifier_and_dedup_validation(self):
        case = cases("A6.1")[0]
        ids = [item["evidence_id"] for item in clinical_media(case)]

        def parse(obj):
            return parse_clinical_answer(json.dumps(obj, ensure_ascii=False), "A6.1")

        valid = {"total": 1, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}]}
        validate_clinical_answer(parse(valid), case)
        self.assertEqual(parse({"total": 0, "images": [{"image": i, "count": 0} for i in ids]})["total"], 0)

        for bad in (
            {"total": True, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}]},
            {"total": 1, "images": [{"image": ids[0], "count": True}, {"image": ids[1], "count": 0}]},
            {"total": 1.0, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}]},
            {"total": -1, "images": [{"image": ids[0], "count": -1}, {"image": ids[1], "count": 0}]},
            {"total": 1, "images": [{"image": ids[0], "count": "1"}, {"image": ids[1], "count": 0}]},
            {"total": 1, "images": [{"image": ids[0], "count": 1}, {"image": ids[0], "count": 0}]},
            {"total": 0, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}]},
            {"total": 3, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}]},
            {"total": 1, "images": [{"image": ids[0], "count": 1}, {"image": ids[1], "count": 0}], "extra": 0},
        ):
            with self.subTest(obj=bad), self.assertRaises(ValueError):
                parse(bad)

        for bad in (
            {"total": 1, "images": [{"image": ids[0], "count": 1}]},
            {"total": 1, "images": [{"image": ids[0], "count": 1}, {"image": "M99", "count": 0}]},
            {"total": 1, "images": [{"image": ids[1], "count": 0}, {"image": ids[0], "count": 1}]},
        ):
            with self.subTest(obj=bad), self.assertRaisesRegex(ValueError, "all input M identifiers"):
                validate_clinical_answer(parse(bad), case)


class A62ScoringTests(unittest.TestCase):
    def test_f1_partial_and_judge_grade_conversion(self):
        ref = {"diagnoses": ["A", "B"], "reason": "医生参考理由"}
        base = score_a62_fields({"diagnoses": ["A"], "reason": "模型理由"}, ref, 100)
        self.assertAlmostEqual(base["field_scores"]["diagnoses"], 100 / 3)
        self.assertIsNone(base["case_score"])
        self.assertEqual(base["pending_review_fields"], ["reason"])

        for grade, reason_points, case_score in ((2, 50, 100), (1, 25, 75), (0, 0, 50)):
            row = copy.deepcopy(base)
            row["task_code"] = "A6.2"
            apply_clinical_judgment(row, {"score": grade, "explanation": "说明"})
            self.assertAlmostEqual(row["field_scores"]["reason"], reason_points)
            self.assertAlmostEqual(row["case_score"], 100 / 3 + reason_points)
            self.assertEqual(row["scoring_status"], "scored")

        self.assertEqual(score_a62_fields(None, ref)["scoring_status"], "invalid")

    def test_option_rejection_reason_length_and_duplicates(self):
        case = cases("A6.2")[0]
        fields = {field["key"]: field["value"] for field in case["input"]["fields"]}
        prompt = build_clinical_prompt(case)
        for option in fields["diagnosis_options"]:
            self.assertIn(option["diagnosis"], prompt)
        option = clinical_reference_answer(case)["diagnoses"][0]
        bad_response = json.dumps({"diagnoses": ["不存在的诊断"], "reason": "测试"}, ensure_ascii=False)
        row = run.evaluate_case(case, bad_response, run.TASK_SPECS["A6.2"]["answer_parser"])
        self.assertIsNone(row["model_answer"])
        self.assertIn("outside input options", row["parse_error"])
        self.assertEqual(row["scoring_status"], "invalid")
        self.assertIsNone(row["case_score"])

        with self.assertRaises(ValueError):
            parse_clinical_answer(
                json.dumps({"diagnoses": [option], "reason": "字" * 101}, ensure_ascii=False), "A6.2"
            )
        with self.assertRaises(ValueError):
            parse_clinical_answer(json.dumps({"diagnoses": [option, option], "reason": "测试"}, ensure_ascii=False), "A6.2")

    def test_judge_prompt_uses_both_diagnoses_and_rejects_bad_schema(self):
        case = cases("A6.2")[0]
        ref = clinical_reference_answer(case)
        messages = build_a62_judge_messages(ref, {"diagnoses": ref["diagnoses"], "reason": "模型理由"})
        content = messages[1]["content"]
        self.assertIn(ref["diagnoses"][0], content)
        self.assertIn(ref["reason"], content)
        self.assertIn("模型理由", content)
        self.assertIn("核心", content)
        self.assertIn("0=", content)
        for bad in ('{"score":3,"explanation":"x"}', '{"score":true,"explanation":"x"}',
                    '{"score":1,"explanation":""}', '{"score":1}'):
            with self.assertRaises(ValueError):
                parse_reason_judgment(bad)


class A71ScoringTests(unittest.TestCase):
    def test_plan_schema_and_pending_state(self):
        case = cases("A7.1")[0]
        valid = {"plans": [{"plan": "牙周基础治疗"}, {"plan": "择期手术"}]}
        parsed = parse_clinical_answer(json.dumps(valid, ensure_ascii=False), "A7.1")
        validate_clinical_answer(parsed, case)
        row = score_a71_fields(parsed, clinical_reference_answer(case))
        self.assertIsNone(row["case_score"])
        self.assertEqual(row["pending_review_fields"], ["plans"])
        self.assertEqual(row["scoring_status"], "pending_review")

        for bad in ({"plans": []}, {"plans": [{"plan": "x", "extra": 1}]}, {"plans": [{"plan": " "}]},
                    {"plans": "x"}, {"plans": [{"plan": 1}]}):
            with self.subTest(obj=bad), self.assertRaises(ValueError):
                parse_clinical_answer(json.dumps(bad, ensure_ascii=False), "A7.1")
        self.assertEqual(score_a71_fields(None, {})["scoring_status"], "invalid")

        with_images = copy.deepcopy(case)
        with_images["input"]["images"] = [{"path": "not-used.jpg"}]
        with self.assertRaisesRegex(ValueError, "text patient records"):
            run.build_openai_messages(with_images, Path("."), build_clinical_prompt)

    def test_holistic_grade_conversion_and_judge_prompt(self):
        case = cases("A7.1")[0]
        ref = clinical_reference_answer(case)
        pred = {"plans": [{"plan": "模型计划一"}, {"plan": "模型计划二"}]}
        base = {"task_code": "A7.1", **score_a71_fields(pred, ref)}
        for grade, expected in ((2, 100), (1, 50), (0, 0)):
            row = copy.deepcopy(base)
            apply_a71_judgment(row, {"score": grade, "explanation": "说明"})
            self.assertEqual(row["case_score"], expected)
            self.assertEqual(row["scoring_status"], "scored")
            self.assertEqual(row["pending_review_fields"], [])

        messages = build_a71_judge_messages(build_clinical_prompt(case), ref["plans"], pred["plans"])
        content = messages[1]["content"]
        self.assertIn("全口牙龈刺激出血数月", content)
        for plan in [item["plan"] for item in ref["plans"] + pred["plans"]]:
            self.assertIn(plan, content)
        self.assertIn("整个模型计划数组", content)
        self.assertIn("数组项数", messages[0]["content"])
        self.assertIn("专家", messages[0]["content"])
        for bad in ('{"score":-1,"explanation":"x"}', '{"score":2,"explanation":1}'):
            with self.assertRaises(ValueError):
                parse_reason_judgment(bad)


class RunnerIntegrationTests(unittest.TestCase):
    def test_a61_never_calls_judge_and_scores_complete(self):
        case = cases("A6.1")[0]
        row, summary, judge_calls, model_calls, _ = execute_task(
            "A6.1", case, clinical_reference_answer(case), judge_reply='{"score":2,"explanation":"不应调用"}'
        )
        self.assertEqual(judge_calls, 0)
        self.assertEqual(model_calls, 1)
        self.assertEqual(row["case_score"], 100)
        self.assertNotIn("reason_judge_model", row)
        self.assertEqual(summary["scoring_status"], "scored")
        self.assertEqual(summary["total_score"], 100)

    def test_a62_runner_one_judge_call_and_failure_keeps_objective_points(self):
        case = cases("A6.2")[0]
        ref = clinical_reference_answer(case)
        pred = {"diagnoses": ref["diagnoses"], "reason": "模型理由"}

        row, summary, judge_calls, _, _ = execute_task(
            "A6.2", case, pred, judge_reply='{"score":2,"explanation":"一致"}'
        )
        self.assertEqual(judge_calls, 1)
        self.assertEqual(row["case_score"], 100)
        self.assertEqual(row["reason_judge_model"], "test-judge")
        self.assertEqual(row["reason_judge_token_usage"]["total_tokens"], 10)
        self.assertEqual(summary["total_score"], 100)

        for reply, error in ((None, RuntimeError("offline")), ("{}", None)):
            row, summary, judge_calls, _, _ = execute_task("A6.2", case, pred, reply, error)
            self.assertEqual(judge_calls, 1)
            self.assertEqual(row["field_scores"]["diagnoses"], 50)
            self.assertIsNone(row["case_score"])
            self.assertEqual(row["pending_review_fields"], ["reason"])
            self.assertTrue(row["reason_judge_error"])
            self.assertIsNone(summary["total_score"])
            self.assertEqual(summary["scoring_status"], "incomplete")

    def test_a71_runner_one_judge_call_and_failure_stays_pending(self):
        case = cases("A7.1")[0]
        pred = {"plans": [{"plan": "牙周基础治疗"}]}
        row, summary, judge_calls, _, _ = execute_task(
            "A7.1", case, pred, judge_reply='{"score":1,"explanation":"部分一致"}'
        )
        self.assertEqual(judge_calls, 1)
        self.assertEqual(row["case_score"], 50)
        self.assertEqual(summary["total_score"], 50)

        row, summary, judge_calls, _, _ = execute_task(
            "A7.1", case, pred, judge_error=RuntimeError("offline")
        )
        self.assertEqual(judge_calls, 1)
        self.assertIsNone(row["case_score"])
        self.assertEqual(row["pending_review_fields"], ["plans"])
        self.assertTrue(row["reason_judge_error"])
        self.assertIsNone(summary["total_score"])
        self.assertEqual(summary["scoring_status"], "incomplete")

    def test_dry_run_never_calls_api_or_writes_results(self):
        for code in ("A6.1", "A6.2", "A7.1"):
            with self.subTest(code=code):
                row, summary, judge_calls, model_calls, output_exists = execute_task(
                    code, cases(code)[0], {}, dry_run=True
                )
                self.assertIsNone(row)
                self.assertIsNone(summary)
                self.assertEqual(judge_calls, 0)
                self.assertEqual(model_calls, 0)
                self.assertFalse(output_exists)


class SummaryAndEvaluateTests(unittest.TestCase):
    def test_a61_complete_partial_and_invalid_summary(self):
        ref = {"total": 2, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 1}]}
        full = {"task_code": "A6.1", "model_answer": copy.deepcopy(ref), **score_a61_fields(copy.deepcopy(ref), ref)}
        partial_pred = {"total": 2, "images": [{"image": "M01", "count": 1}, {"image": "M02", "count": 0}]}
        partial = {"task_code": "A6.1", "model_answer": partial_pred, **score_a61_fields(partial_pred, ref)}
        invalid = {"task_code": "A6.1", "model_answer": None, **score_a61_fields(None, ref)}

        complete = summarize_fields([full], ("total", "images"))
        self.assertEqual(complete["total_score"], 100)
        self.assertEqual(complete["score_rate"], 1.0)
        self.assertEqual(complete["scoring_status"], "scored")

        partial_summary = summarize_fields([partial], ("total", "images"))
        self.assertEqual(partial_summary["total_score"], 75)
        self.assertEqual(partial_summary["objective_metrics"]["images"]["per_image_matched"], 1)
        self.assertEqual(partial_summary["objective_metrics"]["images"]["per_image_total"], 2)
        self.assertEqual(partial_summary["objective_metrics"]["images"]["per_image_accuracy"], 0.5)

        incomplete = summarize_fields([full, invalid], ("total", "images"))
        self.assertIsNone(incomplete["total_score"])
        self.assertEqual(incomplete["scoring_status"], "incomplete")
        self.assertEqual(incomplete["invalid"], 1)
        self.assertEqual(incomplete["invalid_unscored"], 1)
        self.assertEqual(incomplete["fully_scored"], 1)
        self.assertEqual(incomplete["scored_total_score"], 100)

    def test_a62_invalid_is_not_counted_as_zero(self):
        ref = {"diagnoses": ["A"], "reason": "参考"}
        invalid = {"task_code": "A6.2", "model_answer": None, **score_a62_fields(None, ref)}
        summary = summarize_fields([invalid], ("diagnoses", "reason"))
        self.assertIsNone(summary["total_score"])
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(summary["scoring_status"], "incomplete")
        self.assertNotIn("scored_total_score", summary)

    def test_evaluate_reuses_saved_judgment_without_api(self):
        case = cases("A6.2")[0]
        ref = clinical_reference_answer(case)
        pred = {"diagnoses": ref["diagnoses"], "reason": "模型理由"}
        row = {"task_code": "A6.2", "reference_answer": ref, "model_answer": pred,
               "case_points": 100.0, **score_a62_fields(pred, ref)}
        row["task_code"] = "A6.2"
        apply_clinical_judgment(row, {"score": 2, "explanation": "一致"})

        with patch.object(run, "chat_completion", side_effect=AssertionError("evaluate must not call the API")):
            recomputed = evaluate.recompute_clinical_row(row)
        self.assertEqual(recomputed["case_score"], 100)
        self.assertEqual(recomputed["scoring_status"], "scored")
        self.assertEqual(recomputed["reason_judgment"]["score"], 2)

    def test_evaluate_keeps_legacy_completed_score_when_judgment_is_absent(self):
        ref = {"diagnoses": ["A"], "reason": "参考"}
        pred = {"diagnoses": ["A"], "reason": "模型理由"}
        row = {"task_code": "A6.2", "reference_answer": ref, "model_answer": pred,
               "case_points": 100.0, **score_a62_fields(pred, ref)}
        row["task_code"] = "A6.2"
        row.update({"scoring_status": "scored", "case_score": 75.0,
                    "field_scores": {"diagnoses": 50.0, "reason": 25.0}, "pending_review_fields": []})
        recomputed = evaluate.recompute_clinical_row(row)
        self.assertEqual(recomputed["case_score"], 75.0)
        self.assertEqual(recomputed["scoring_status"], "scored")
        self.assertEqual(recomputed["field_scores"]["reason"], 25.0)

    def test_evaluate_preserves_existing_a5_results(self):
        for code in ("A5.1", "A5.2", "A5.3"):
            row = {"task_code": code, "reference_answer": {}, "case_score": 37.5,
                   "scoring_status": "scored", "reason_judgment": {"legacy": True}}
            self.assertIs(evaluate.recompute_clinical_row(row), row)

    def test_evaluate_rejects_invalid_saved_judge_grade(self):
        case = cases("A7.1")[0]
        ref = clinical_reference_answer(case)
        pred = {"plans": [{"plan": "测试"}]}
        row = {"task_code": "A7.1", "reference_answer": ref, "model_answer": pred,
               **score_a71_fields(pred, ref),
               "reason_judgment": {"score": True, "explanation": "不能当整数处理"}}
        recomputed = evaluate.recompute_clinical_row(row)
        self.assertIsNone(recomputed["case_score"])
        self.assertEqual(recomputed["scoring_status"], "pending_review")

    def test_evaluate_reuses_saved_a71_judgment(self):
        case = cases("A7.1")[0]
        ref = clinical_reference_answer(case)
        pred = {"plans": [{"plan": "牙周基础治疗"}]}
        row = {"task_code": "A7.1", "reference_answer": ref, "model_answer": pred,
               "case_points": 100.0, **score_a71_fields(pred, ref)}
        apply_a71_judgment(row, {"score": 1, "explanation": "部分"})
        recomputed = evaluate.recompute_clinical_row(row)
        self.assertEqual(recomputed["case_score"], 50)
        self.assertEqual(recomputed["scoring_status"], "scored")


class PromptIsolationTests(unittest.TestCase):
    def test_media_order_follows_sent_images_for_a62(self):
        case = copy.deepcopy(cases("A6.2")[0])
        expected = [item["evidence_id"] for item in clinical_media(case)]
        for field in case["input"]["fields"]:
            if field["key"] == "evidence_media":
                field["value"].reverse()
        self.assertEqual([item["evidence_id"] for item in clinical_media(case)], expected)

    def test_reference_answers_never_enter_evaluated_prompts(self):
        for code in ("A6.1", "A6.2", "A7.1"):
            with self.subTest(code=code):
                case = cases(code)[0]
                prompt = build_clinical_prompt(case)
                changed = copy.deepcopy(case)
                changed["reference_answer"] = {"fields": [{"key": "secret", "value": "PRIVATE_REFERENCE"}]}
                self.assertEqual(build_clinical_prompt(changed), prompt)
                self.assertNotIn("PRIVATE_REFERENCE", prompt)


if __name__ == "__main__":
    unittest.main()
