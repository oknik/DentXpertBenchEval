"""Dynamic prompt contexts and template selection for A1-A4."""

import json
from typing import Any

from bench import protocol
from bench.prompt_renderer import render_prompt_file


def build_a11_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A1.1", {})


def build_a12_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A1.2", {})


def build_a13_prompt(case: dict[str, Any]) -> str:
    purpose = next(item["value"] for item in case["input"].get("fields", []) if item.get("key") == "specified_purpose")
    return render_prompt_file("A1.3", {"clinical_purpose": protocol.format_clinical_purpose(purpose)})


def build_a21_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A2.1", {})


def build_a22_prompt(case: dict[str, Any]) -> str:
    mode = protocol.get_field_value(case, "localization_mode")
    if mode == "inventory":
        return render_prompt_file("A2.2", {}, variant="inventory")
    if mode == "detection":
        targets = json.dumps(protocol.extract_a22_reference_fdis(case), ensure_ascii=False)
        return render_prompt_file("A2.2", {"target_fdis": targets}, variant="detection")
    raise ValueError(f"unsupported A2.2 localization_mode: {mode}")


def build_a23_prompt(case: dict[str, Any]) -> str:
    targets = protocol.extract_a23_target_fdis(case)
    text = json.dumps(targets, ensure_ascii=False) if targets else "{{题目给定的FDI整数数组}}"
    return render_prompt_file("A2.3", {"target_fdis": text})


def build_a24_prompt(case: dict[str, Any]) -> str:
    modality = str(case.get("input", {}).get("images", [{}])[0].get("modality", "")).strip()
    if modality == "cephalometric_radiograph":
        structures = protocol.select_a24_cephalometric_structures(case)
        text = "\n".join(protocol.describe_a24_cephalometric_structure(code) for code in structures)
        return render_prompt_file("A2.4", {"target_structures": text}, variant="cephalometric")
    if modality == "periapical_radiograph":
        return render_prompt_file("A2.4", {}, variant="periapical")
    raise ValueError(f"unsupported A2.4 modality: {modality}")


def _values(categories: list[str], scope: str, options: list[str]) -> dict[str, str]:
    return {"categories": json.dumps(categories, ensure_ascii=False) if categories else "未指定", "scope": scope, "options": "、".join(options)}


def build_a31_prompt(case: dict[str, Any]) -> str:
    scope = protocol.get_a31_scope(case)
    mode = protocol.get_a31_mode(case)
    if scope["site_type"] == "tooth" and scope["fdis"]:
        scope_text = json.dumps(scope["fdis"], ensure_ascii=False)
    elif scope["site_type"] == "region" and scope["regions"]:
        scope_text = json.dumps(scope["regions"], ensure_ascii=False)
    else:
        scope_text = "全口"
    return render_prompt_file("A3.1", _values(scope["concepts"], scope_text, protocol.select_a31_prompt_options(case)), variant=mode)


def build_a32_prompt(case: dict[str, Any]) -> str:
    scope = protocol.get_a32_scope(case)
    scope_text = "全口" if scope["mode"] == "region" else (json.dumps(scope["regions"], ensure_ascii=False) if scope["regions"] else "全口")
    return render_prompt_file("A3.2", _values(scope["concepts"], scope_text, protocol.select_a32_prompt_options(case)), variant=scope["mode"])


def _mixed_scope(scope: dict[str, Any]) -> str:
    if scope["mode"] == "region":
        return "全口"
    if scope["site_type"] == "tooth":
        return json.dumps(scope["fdis"], ensure_ascii=False) if scope["fdis"] else "全口"
    if scope["site_type"] == "region":
        return json.dumps(scope["regions"], ensure_ascii=False) if scope["regions"] else "全口"
    parts = [json.dumps(scope[key], ensure_ascii=False) for key in ("fdis", "regions") if scope[key]]
    return "、".join(parts) if parts else "全口"


def _render_mixed_localization(task_code: str, mode: str, values: dict[str, str]) -> str:
    header = render_prompt_file(task_code, {**values, "mode_instructions": ""})
    instructions = render_prompt_file(task_code, {}, variant=mode)
    return f"{header}\n\n{instructions}"


def build_a33_prompt(case: dict[str, Any]) -> str:
    scope = protocol.get_a33_scope(case)
    values = _values(scope["concepts"], _mixed_scope(scope), protocol.select_a33_prompt_options(case))
    return _render_mixed_localization("A3.3", scope["mode"], values)


def build_a34_prompt(case: dict[str, Any]) -> str:
    scope = protocol.get_a34_scope(case)
    values = _values(scope["concepts"], _mixed_scope(scope), protocol.select_a34_prompt_options(case))
    return _render_mixed_localization("A3.4", scope["mode"], values)


def build_a41_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A4.1", {})


def build_a42_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A4.2", {})


def build_a43_prompt(case: dict[str, Any]) -> str:
    return render_prompt_file("A4.3", {"target_surfaces": "、".join(protocol.get_a43_target_surfaces(case))})
