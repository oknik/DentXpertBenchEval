"""Metrics for geometric localization tasks.

This module owns all scoring that compares image coordinates: bounding boxes,
IoU, and keypoints.  Classification metrics stay in ``classification.py``.
"""

import math
from typing import Any

REGION_EQUIVALENT_CHILDREN = {
    "前牙区": ("上颌前牙区", "下颌前牙区"),
    "左侧后牙区": ("左上后牙区", "左下后牙区"),
    "右侧后牙区": ("右上后牙区", "右下后牙区"),
}

# A2.4 periapical periodontal annotation no longer includes root-apex
# landmarks. Keep accepting legacy answers, but exclude this obsolete point
# from both sides of the comparison so it cannot affect the score.
IGNORED_A24_STRUCTURE_CODES = {"ROOT_APEX"}


def score_a22_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch A2.2 inventory or bounding-box scoring by reference shape."""
    from bench.metrics.classification import score_a22_inventory_fields

    ref_teeth = ref.get("teeth", [])
    if ref_teeth and isinstance(ref_teeth[0], dict):
        return score_a22_detection_fields(pred, ref, case_points, scoring)
    return score_a22_inventory_fields(pred, ref, case_points, scoring)


def score_a22_detection_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score tooth bounding boxes by FDI identity and IoU."""
    pred_by_fdi = {item["fdi"]: item["bbox"] for item in pred.get("teeth", [])} if pred else {}
    ref_by_fdi = {item["fdi"]: item["bbox"] for item in ref.get("teeth", [])}

    ious = {}
    for fdi, ref_bbox in ref_by_fdi.items():
        pred_bbox = pred_by_fdi.get(fdi)
        ious[fdi] = bbox_iou(pred_bbox, ref_bbox) if pred_bbox else 0.0

    expected_fdis = set(ref_by_fdi)
    predicted_fdis = set(pred_by_fdi)
    fdi_set_ok = predicted_fdis == expected_fdis
    mean_iou = sum(ious.values()) / len(ref_by_fdi) if ref_by_fdi else 0.0
    extra_fdis = predicted_fdis - expected_fdis
    weight = scoring_field_weight(scoring, "teeth")
    case_score = case_points * weight * mean_iou if not extra_fdis else 0.0

    return {
        "field_correct": {"teeth": fdi_set_ok and mean_iou >= 0.5},
        "field_scores": {"teeth": case_score},
        "correct_fields": int(fdi_set_ok and mean_iou >= 0.5),
        "total_fields": 1,
        "case_correct": fdi_set_ok and all(value >= 0.5 for value in ious.values()),
        "case_points": case_points,
        "case_score": case_score,
        "tooth_metrics": {
            "mode": "detection",
            "mean_iou": mean_iou,
            "per_fdi_iou": ious,
            "missing_fdis": sorted(expected_fdis - predicted_fdis),
            "extra_fdis": sorted(extra_fdis),
        },
    }


def score_a24_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score cephalometric/periapical keypoints by normalized distance."""
    pred_by_key = {
        structure_key(item): item["coordinate"]
        for item in (pred.get("structures", []) if pred else [])
        if item.get("structure_code") not in IGNORED_A24_STRUCTURE_CODES
    }
    ref_by_key = {
        structure_key(item): item["coordinate"]
        for item in ref.get("structures", [])
        if item.get("structure_code") not in IGNORED_A24_STRUCTURE_CODES
    }

    distances = {}
    tolerance = scoring_field_parameter(scoring, "distance_tolerance", 0.05)
    per_structure_score = {}
    for key, ref_coordinate in ref_by_key.items():
        pred_coordinate = pred_by_key.get(key)
        distance = coordinate_distance(pred_coordinate, ref_coordinate) if pred_coordinate else None
        distances[format_structure_key(key)] = distance
        per_structure_score[format_structure_key(key)] = coordinate_score(
            distance,
            tolerance,
        )

    expected_keys = set(ref_by_key)
    predicted_keys = set(pred_by_key)
    missing = sorted(expected_keys - predicted_keys, key=format_structure_key)
    extra = sorted(predicted_keys - expected_keys, key=format_structure_key)
    matched_keys = expected_keys & predicted_keys
    mean_coordinate_score = sum(per_structure_score.values()) / len(ref_by_key) if ref_by_key else 0.0
    precision = len(matched_keys) / len(predicted_keys) if predicted_keys else 0.0
    recall = len(matched_keys) / len(expected_keys) if expected_keys else (1.0 if not predicted_keys else 0.0)
    key_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 1.0
    union_keys = expected_keys | predicted_keys
    union_score = sum(per_structure_score.get(format_structure_key(key), 0.0) for key in union_keys) / len(union_keys) if union_keys else 1.0
    weight = scoring_field_weight(scoring, "structures")
    exact_key_set = predicted_keys == expected_keys
    all_close = exact_key_set and all(
        distance is not None and distance <= tolerance for distance in distances.values()
    )

    return {
        "field_correct": {"structures": all_close},
        "field_scores": {"structures": case_points * weight * union_score},
        "correct_fields": int(all_close),
        "total_fields": 1,
        "case_correct": all_close,
        "case_points": case_points,
        "case_score": case_points * weight * union_score,
        "structure_metrics": {
            "mean_score": union_score,
            "union_score": union_score,
            "union_structure_count": len(union_keys),
            "mean_coordinate_score": mean_coordinate_score,
            "precision": precision,
            "recall": recall,
            "key_f1": key_f1,
            "matched_structures": [format_structure_key(key) for key in sorted(matched_keys, key=format_structure_key)],
            "per_structure_distance": distances,
            "missing_structures": [format_structure_key(key) for key in missing],
            "extra_structures": [format_structure_key(key) for key in extra],
        },
    }


def score_a31_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score A3.1 hard-tissue findings according to the reference localization shape."""
    ref_items = ref.get("findings", [])
    pred_items = pred.get("findings", []) if pred else []
    if any("bbox" in item for item in ref_items):
        return score_a31_bbox_fields(pred_items, ref_items, case_points, scoring)
    if any("region_code" in item for item in ref_items):
        return score_a31_set_fields(pred_items, ref_items, "region_code", case_points, scoring, "region")
    return score_a31_tooth_fields(pred_items, ref_items, case_points, scoring)


def score_a31_tooth_fields(pred: list[dict[str, Any]], ref: list[dict[str, Any]], case_points: float, scoring: dict[str, Any] | None) -> dict[str, Any]:
    def expand(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
        return {(item["finding_code"], fdi) for item in items for fdi in item.get("fdi", [])}
    return score_a31_set_fields(list(expand(pred)), list(expand(ref)), None, case_points, scoring, "tooth")


def score_a31_set_fields(
    pred: list[Any],
    ref: list[Any],
    key: str | None,
    case_points: float,
    scoring: dict[str, Any] | None,
    mode: str,
    region_equivalence: bool = False,
) -> dict[str, Any]:
    def token(item: Any) -> tuple[Any, ...]:
        if isinstance(item, tuple):
            return item
        return (item["finding_code"], item[key])
    def expand_region_tokens(items: list[Any], expand_parents: bool) -> set[tuple[Any, ...]]:
        tokens: set[tuple[Any, ...]] = set()
        for item in items:
            current = token(item)
            if region_equivalence and mode == "region" and expand_parents and len(current) == 2:
                finding_code, region_code = current
                children = REGION_EQUIVALENT_CHILDREN.get(region_code)
                if children:
                    tokens.update((finding_code, child) for child in children)
                    continue
            tokens.add(current)
        return tokens

    raw_predicted = {token(item) for item in pred}
    raw_expected = {token(item) for item in ref}
    if region_equivalence and mode == "region" and raw_predicted == raw_expected:
        predicted, expected = raw_predicted, raw_expected
    else:
        # The reference may use a broad parent region and the model may answer
        # with both child regions. Do not apply this expansion in reverse:
        # a broad prediction must not receive partial credit for a narrow reference.
        predicted = expand_region_tokens(pred, expand_parents=False)
        expected = expand_region_tokens(ref, expand_parents=True)
    intersection = predicted & expected
    precision = len(intersection) / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = len(intersection) / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 1.0
    union = predicted | expected
    union_score = len(intersection) / len(union) if union else 1.0
    exact = predicted == expected
    return {
        "field_correct": {"findings": exact},
        "field_scores": {"findings": case_points * scoring_field_weight(scoring, "findings") * union_score},
        "correct_fields": int(exact), "total_fields": 1,
        "case_correct": exact, "case_points": case_points,
        "case_score": case_points * scoring_field_weight(scoring, "findings") * union_score,
        "finding_metrics": {"mode": mode, "precision": precision, "recall": recall, "f1": f1,
                            "missing": sorted(expected - predicted, key=str), "extra": sorted(predicted - expected, key=str),
                            "correct_count": len(intersection), "union_count": len(union),
                            "union_score": union_score},
    }


def score_a31_bbox_fields(pred: list[dict[str, Any]], ref: list[dict[str, Any]], case_points: float, scoring: dict[str, Any] | None) -> dict[str, Any]:
    threshold = scoring_field_parameter(scoring, "iou_threshold", 0.5)
    valid_candidates: dict[int, list[tuple[int, float, float]]] = {}
    for reference_index, expected in enumerate(ref):
        valid_candidates[reference_index] = []
        for prediction_index, item in enumerate(pred):
            if item["finding_code"] != expected["finding_code"]:
                continue
            iou = bbox_iou(item["bbox"], expected["bbox"])
            if iou >= threshold:
                valid_candidates[reference_index].append(
                    (prediction_index, iou, bbox_center_distance(item["bbox"], expected["bbox"]))
                )

    # Find a one-to-one assignment that first maximizes the number of valid
    # matches, then minimizes total center distance. This prevents a single
    # prediction from being consumed by an earlier reference merely because
    # it was the first item in the reference list.
    def find_assignment(
        reference_index: int,
        used_predictions: frozenset[int],
    ) -> tuple[tuple[tuple[int, int, float, float], ...], int, float]:
        if reference_index >= len(ref):
            return (), 0, 0.0

        best_assignment, best_count, best_distance = find_assignment(
            reference_index + 1, used_predictions
        )
        for prediction_index, iou, center_distance in valid_candidates[reference_index]:
            if prediction_index in used_predictions:
                continue
            tail_assignment, tail_count, tail_distance = find_assignment(
                reference_index + 1, used_predictions | {prediction_index}
            )
            candidate_assignment = ((reference_index, prediction_index, iou, center_distance),) + tail_assignment
            candidate_count = tail_count + 1
            candidate_distance = tail_distance + center_distance
            if (
                candidate_count > best_count
                or (candidate_count == best_count and candidate_distance < best_distance)
            ):
                best_assignment, best_count, best_distance = (
                    candidate_assignment,
                    candidate_count,
                    candidate_distance,
                )
        return best_assignment, best_count, best_distance

    assignment, correct_matches, _ = find_assignment(0, frozenset())
    assignment_by_reference = {item[0]: item for item in assignment}
    ious = [assignment_by_reference[index][2] if index in assignment_by_reference else 0.0 for index in range(len(ref))]
    center_distances = [
        assignment_by_reference[index][3] if index in assignment_by_reference else float("inf")
        for index in range(len(ref))
    ]
    matched_pairs = [
        {
            "reference_index": reference_index,
            "prediction_index": prediction_index,
            "finding_code": ref[reference_index]["finding_code"],
            "center_distance": center_distance,
            "iou": iou,
        }
        for reference_index, prediction_index, iou, center_distance in assignment
    ]
    union_count = len(ref) + len(pred) - correct_matches
    union_score = (
        sum(iou for iou in ious if iou >= threshold) / union_count
        if union_count
        else 1.0
    )
    exact = len(pred) == len(ref) == correct_matches
    score = case_points * scoring_field_weight(scoring, "findings") * union_score
    return {
        "field_correct": {"findings": exact}, "field_scores": {"findings": score},
        "correct_fields": int(exact), "total_fields": 1, "case_correct": exact,
        "case_points": case_points, "case_score": score,
        "finding_metrics": {"mode": "bbox", "per_finding_iou": ious,
                            "center_distances": center_distances, "matched_pairs": matched_pairs,
                            "correct_count": correct_matches, "union_count": union_count,
                            "union_score": union_score,
                            "missing": max(0, len(ref) - correct_matches),
                            "extra": max(0, len(pred) - correct_matches),
                            "iou_threshold": threshold},
    }


def score_a32_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score A3.2 soft-tissue findings in region or bbox localization mode."""
    pred_items = pred.get("findings", []) if pred else []
    ref_items = ref.get("findings", [])

    def to_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
        bbox = item.get("bbox", item.get("normalized_bbox"))
        return {"finding_code": item["finding_code"], "bbox": bbox} if bbox is not None else None

    has_bbox = any(to_bbox(item) is not None for item in [*ref_items, *pred_items])
    if has_bbox:
        normalized_ref = [to_bbox(item) for item in ref_items]
        normalized_pred = [to_bbox(item) for item in pred_items]
        return score_a31_bbox_fields(
            [item for item in normalized_pred if item is not None],
            [item for item in normalized_ref if item is not None],
            case_points,
            scoring,
        )

    def expand(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
        expanded: list[tuple[str, str]] = []
        for item in items:
            site = item.get("site") if isinstance(item.get("site"), dict) else {}
            regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
            if isinstance(regions, str):
                regions = [regions]
            expanded.extend((item["finding_code"], region) for region in regions)
        return expanded

    return score_a31_set_fields(
        expand(pred_items),
        expand(ref_items),
        None,
        case_points,
        scoring,
        "region",
    )


def score_a33_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score A3.3 orthodontic/developmental findings by tooth, region, or bbox."""
    pred_items = pred.get("findings", []) if pred else []
    ref_items = ref.get("findings", [])

    def to_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
        bbox = item.get("bbox", item.get("normalized_bbox"))
        return {"finding_code": item["finding_code"], "bbox": bbox} if bbox is not None else None

    if any(to_bbox(item) is not None for item in [*ref_items, *pred_items]):
        normalized_ref = [to_bbox(item) for item in ref_items]
        normalized_pred = [to_bbox(item) for item in pred_items]
        return score_a31_bbox_fields(
            [item for item in normalized_pred if item is not None],
            [item for item in normalized_ref if item is not None],
            case_points,
            scoring,
        )

    def expand(items: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        expanded: list[tuple[str, Any]] = []
        for item in items:
            site = item.get("site") if isinstance(item.get("site"), dict) else {}
            if "fdi" in item:
                expanded.extend((item["finding_code"], fdi) for fdi in item["fdi"])
            elif site.get("type") == "tooth":
                expanded.extend((item["finding_code"], int(fdi)) for fdi in site.get("fdi", []))
            else:
                regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
                if isinstance(regions, str):
                    regions = [regions]
                expanded.extend((item["finding_code"], region) for region in regions)
        return expanded

    mode = "tooth" if any("fdi" in item or (isinstance(item.get("site"), dict) and item["site"].get("type") == "tooth") for item in [*ref_items, *pred_items]) else "region"
    return score_a31_set_fields(
        expand(pred_items), expand(ref_items), None, case_points, scoring, mode,
        region_equivalence=True,
    )


def score_a34_fields(
    pred: dict[str, Any] | None,
    ref: dict[str, Any],
    case_points: float = 1.0,
    scoring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score A3.4 treatment/restoration findings by tooth, region, or bbox."""
    pred_items = pred.get("findings", []) if pred else []
    ref_items = ref.get("findings", [])

    def to_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
        bbox = item.get("bbox", item.get("normalized_bbox"))
        return {"finding_code": item["finding_code"], "bbox": bbox} if bbox is not None else None

    if any(to_bbox(item) is not None for item in [*ref_items, *pred_items]):
        normalized_ref = [to_bbox(item) for item in ref_items]
        normalized_pred = [to_bbox(item) for item in pred_items]
        return score_a31_bbox_fields(
            [item for item in normalized_pred if item is not None],
            [item for item in normalized_ref if item is not None],
            case_points,
            scoring,
        )

    def expand(items: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        expanded: list[tuple[str, Any]] = []
        for item in items:
            if "fdi" in item:
                expanded.extend((item["finding_code"], fdi) for fdi in item["fdi"])
            else:
                site = item.get("site") if isinstance(item.get("site"), dict) else {}
                if site.get("type") == "tooth":
                    expanded.extend(
                        (item["finding_code"], fdi if str(fdi).lower() == "x" else int(fdi))
                        for fdi in site.get("fdi", [])
                    )
                else:
                    regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
                    if isinstance(regions, str):
                        regions = [regions]
                    expanded.extend((item["finding_code"], region) for region in regions)
        return expanded

    mode = "tooth" if any("fdi" in item or (isinstance(item.get("site"), dict) and item["site"].get("type") == "tooth") for item in [*ref_items, *pred_items]) else "region"
    return score_a31_set_fields(
        expand(pred_items), expand(ref_items), None, case_points, scoring, mode,
        region_equivalence=True,
    )


def structure_key(item: dict[str, Any]) -> tuple[str, int | None]:
    return str(item["structure_code"]), item.get("fdi")


def format_structure_key(key: tuple[str, int | None]) -> str:
    structure_code, fdi = key
    return f"{fdi}:{structure_code}" if fdi is not None else structure_code


def coordinate_distance(a: list[float] | None, b: list[float]) -> float:
    if not a:
        return 1.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def coordinate_score(distance: float | None, tolerance: float = 0.05) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - distance / tolerance)


def bbox_iou(a: list[float] | None, b: list[float]) -> float:
    if not a:
        return 0.0
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def bbox_center_distance(a: list[float], b: list[float]) -> float:
    """Return normalized Euclidean distance between two bbox centers."""
    a_center = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    b_center = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return math.dist(a_center, b_center)


def scoring_field_config(scoring: dict[str, Any] | None, field_key: str) -> dict[str, Any]:
    for item in (scoring or {}).get("fields", []):
        if isinstance(item, dict) and item.get("key") == field_key:
            return item
    return {}


def scoring_field_weight(scoring: dict[str, Any] | None, field_key: str) -> float:
    return float(scoring_field_config(scoring, field_key).get("weight", 1.0))


def scoring_field_parameter(scoring: dict[str, Any] | None, parameter: str, default: float) -> float:
    fields = (scoring or {}).get("fields", [])
    for item in fields:
        if isinstance(item, dict) and parameter in item:
            return float(item[parameter])
    return default

