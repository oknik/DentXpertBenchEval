import copy
import json
import unittest

from bench.metrics.clinical import apply_reason_judgment, build_reason_judge_messages, parse_reason_judgment, score_a51_fields
from bench.metrics.report import summarize_fields
from bench.protocol import parse_clinical_answer


class A51ScoringTests(unittest.TestCase):
    def test_f1_partial_and_judge_point_conversion(self):
        ref = {"diagnoses": ["A", "B"], "reason": "参考"}
        row = score_a51_fields({"diagnoses": ["A"], "reason": "预测"}, ref, 100)
        self.assertAlmostEqual(row["field_scores"]["diagnoses"], 100 / 3)
        self.assertIsNone(row["case_score"])
        apply_reason_judgment(row, {"score": 1, "explanation": "部分一致"})
        self.assertEqual(row["field_scores"]["reason"], 25)
        self.assertAlmostEqual(row["case_score"], 100 / 3 + 25)
        self.assertEqual(row["scoring_status"], "scored")
        wrong = score_a51_fields({"diagnoses": ["C"], "reason": "预测"}, ref)
        self.assertEqual(wrong["field_scores"]["diagnoses"], 0)

    def test_judge_validation_and_prompt(self):
        messages = build_reason_judge_messages("参考依据", "不同表述")
        self.assertIn("参考依据", messages[1]["content"])
        self.assertIn("不同表述", messages[1]["content"])
        for invalid in ('{"score":3,"explanation":"x"}', '{"score":true,"explanation":"x"}', '{"score":1,"explanation":""}'):
            with self.assertRaises(ValueError):
                parse_reason_judgment(invalid)
        self.assertEqual(parse_reason_judgment('{"score":2,"explanation":"一致"}')["score"], 2)
        with self.assertRaises(ValueError):
            parse_clinical_answer(json.dumps({"diagnoses": ["A"], "reason": "字" * 101}, ensure_ascii=False), "A5.1")

    def test_unscored_cases_do_not_silently_count_as_zero(self):
        ref = {"diagnoses": ["A"], "reason": "参考"}
        scored = {"task_code": "A5.1", "model_answer": {"diagnoses": ["A"]}, **score_a51_fields({"diagnoses": ["A"]}, ref, 100)}
        apply_reason_judgment(scored, {"score": 2, "explanation": "一致"})
        pending = copy.deepcopy(scored)
        pending.update({"case_score": None, "scoring_status": "pending_review", "pending_review_fields": ["reason"]})
        summary = summarize_fields([scored, pending], ("diagnoses", "reason"))
        self.assertIsNone(summary["total_score"])
        self.assertEqual(summary["scored_average_score"], 100)
        self.assertEqual(summary["unscored"], 1)


if __name__ == "__main__":
    unittest.main()
