import copy
import json
import unittest

import run
from bench.metrics.clinical import apply_clinical_judgment, build_a52_judge_messages, parse_a52_judgment, score_a52_fields, score_a53_fields
from bench.protocol import build_clinical_prompt, clinical_diagnosis_options, clinical_reference_answer, parse_clinical_answer, validate_clinical_answer


def cases(code):
    return json.loads(run.TASK_SPECS[code]["case_path"].read_text(encoding="utf-8"))["cases"]


class A52A53Tests(unittest.TestCase):
    def test_all_a52_candidate_sets_and_private_labels(self):
        for case in cases("A5.2"):
            with self.subTest(case=case["id"]):
                fields = {f["key"]: f["value"] for f in case["input"]["fields"]}
                entries = case["reference_answer"]["fields"][0]["value"]
                ids = {item["option_id"] for item in entries}
                expected = [item for item in fields["diagnosis_options"] if item["option_id"] in ids]
                self.assertEqual(clinical_diagnosis_options(case), expected)
                prompt = build_clinical_prompt(case)
                option_block = prompt.split("诊断选项：\n", 1)[1].split("\n\n判定与填写规则", 1)[0]
                self.assertEqual(option_block, "、".join(item["diagnosis"] for item in expected))
                changed = copy.deepcopy(case)
                for entry in changed["reference_answer"]["fields"][0]["value"]:
                    entry.update(role="PRIVATE_ROLE", reason="PRIVATE_REASON")
                self.assertEqual(build_clinical_prompt(changed), prompt)
                hidden = next(item for item in fields["diagnosis_options"] if item["option_id"] not in ids)
                with self.assertRaisesRegex(ValueError, "outside input options"):
                    validate_clinical_answer({"diagnoses": [hidden["diagnosis"]]}, case)

    def test_bad_candidate_ids_fail_explicitly(self):
        case = copy.deepcopy(cases("A5.2")[0])
        case["reference_answer"]["fields"][0]["value"][0]["option_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "candidate IDs"):
            build_clinical_prompt(case)

    def test_a52_scores_and_joint_judge_schema(self):
        row = {"task_code": "A5.2", **score_a52_fields({"diagnoses": ["A"]}, {"diagnoses": ["A", "B"]})}
        self.assertAlmostEqual(row["field_scores"]["diagnoses"], 100 / 3)
        judgment = parse_a52_judgment('{"reason":{"score":2,"explanation":"一致"},"exclude_reason":{"score":1,"explanation":"部分"}}')
        apply_clinical_judgment(row, judgment)
        self.assertEqual(row["field_scores"]["reason"], 25)
        self.assertEqual(row["field_scores"]["exclude_reason"], 12.5)
        self.assertAlmostEqual(row["case_score"], 100 / 3 + 37.5)
        for response in ('{}', '{"reason":{"score":2,"explanation":"x"}}', '{"reason":{"score":2,"explanation":"x"},"exclude_reason":{"score":true,"explanation":"x"}}'):
            with self.assertRaises(ValueError):
                parse_a52_judgment(response)
        ref = clinical_reference_answer(cases("A5.2")[0])
        messages = build_a52_judge_messages(ref, {"diagnoses": ref["diagnoses"], "reason": "预测", "exclude_reason": "排除"})
        self.assertIn(ref["exclude_reason"], messages[1]["content"])

    def test_a53_mapping_and_all_conditional_scores(self):
        for case in cases("A5.3"):
            ref = clinical_reference_answer(case)
            expected = "充足" if ref["diagnosis_support"] == "supported" else "不足"
            self.assertEqual(ref["sufficiency"], expected)
            self.assertTrue(ref["reason"])
            pred = {"sufficiency": expected}
            if expected == "不足":
                pred["reason"] = "理由"
            row = {"task_code": "A5.3", **score_a53_fields(pred, ref)}
            if expected == "充足":
                self.assertEqual(row["case_score"], 100)
                self.assertEqual(row["pending_review_fields"], [])
            else:
                self.assertEqual(row["field_scores"]["sufficiency"], 50)
                apply_clinical_judgment(row, {"score": 1, "explanation": "部分"})
                self.assertEqual(row["case_score"], 75)
            wrong = score_a53_fields({"sufficiency": "不足" if expected == "充足" else "充足"}, ref)
            self.assertEqual(wrong["case_score"], 0)
            self.assertEqual(wrong["pending_review_fields"], [])
        parsed = parse_clinical_answer('{"sufficiency":"不充足","reason":"理由"}', "A5.3")
        self.assertEqual(parsed["sufficiency"], "不足")
        with self.assertRaises(ValueError):
            parse_clinical_answer('{"sufficiency":"充足","reason":"多余"}', "A5.3")
        with self.assertRaises(ValueError):
            parse_clinical_answer('{"sufficiency":"不足"}', "A5.3")
        with self.assertRaises(ValueError):
            parse_clinical_answer(json.dumps({"sufficiency": "不足", "reason": "字" * 101}), "A5.3")


if __name__ == "__main__":
    unittest.main()
