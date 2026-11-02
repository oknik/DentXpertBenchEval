import json
import hashlib
import random
import re
from typing import Any

from bench.prompt_renderer import render_prompt_file

A11_FIELD_KEYS = ("modality_type", "field_of_view")
A11_ANSWER_KEYS = A11_FIELD_KEYS
A12_FIELD_KEYS = ("coverage_codes", "quality_grade")
A12_ANSWER_KEYS = A12_FIELD_KEYS
A13_FIELD_KEYS = ("sufficiency", "minimum_necessary_action")
A13_ANSWER_KEYS = A13_FIELD_KEYS
A21_FIELD_KEYS = ("dentition_stage",)
A21_ANSWER_KEYS = A21_FIELD_KEYS
A22_FIELD_KEYS = ("teeth",)
A22_ANSWER_KEYS = A22_FIELD_KEYS
A23_FIELD_KEYS = ("teeth",)
A23_ANSWER_KEYS = A23_FIELD_KEYS
A24_FIELD_KEYS = ("structures",)
A31_FIELD_KEYS = ("findings",)
A31_ANSWER_KEYS = A31_FIELD_KEYS
A24_ANSWER_KEYS = A24_FIELD_KEYS
A32_FIELD_KEYS = ("findings",)
A32_ANSWER_KEYS = A32_FIELD_KEYS
A33_FIELD_KEYS = ("findings",)
A33_ANSWER_KEYS = A33_FIELD_KEYS
A34_FIELD_KEYS = ("findings",)
A34_ANSWER_KEYS = A34_FIELD_KEYS
A41_FIELD_KEYS = ("grade_value",)
A41_ANSWER_KEYS = A41_FIELD_KEYS
A42_FIELD_KEYS = ("grade_value",)
A42_ANSWER_KEYS = A42_FIELD_KEYS
A43_FIELD_KEYS = ("surface_grades",)
A43_ANSWER_KEYS = A43_FIELD_KEYS

CLINICAL_FIELD_KEYS = {
    "A5.1": ("diagnoses", "reason"),
    "A5.2": ("diagnoses", "reason", "exclude_reason"),
    "A5.3": ("sufficiency", "reason"),
    "A6.1": ("total", "images"),
    "A6.2": ("diagnoses", "reason"),
    "A6.3": ("category", "condition_code"),
    "A7.1": ("plans",),
}

# Keep one source of truth for the public task-to-field contract.  The runner,
# evaluator, and reporting code all consume this mapping instead of repeating
# the A1-A4 and A5-A7 lists independently.
TASK_FIELD_KEYS = {
    "A1.1": A11_FIELD_KEYS,
    "A1.2": A12_FIELD_KEYS,
    "A1.3": A13_FIELD_KEYS,
    "A2.1": A21_FIELD_KEYS,
    "A2.2": A22_FIELD_KEYS,
    "A2.3": A23_FIELD_KEYS,
    "A2.4": A24_FIELD_KEYS,
    "A3.1": A31_FIELD_KEYS,
    "A3.2": A32_FIELD_KEYS,
    "A3.3": A33_FIELD_KEYS,
    "A3.4": A34_FIELD_KEYS,
    "A4.1": A41_FIELD_KEYS,
    "A4.2": A42_FIELD_KEYS,
    "A4.3": A43_FIELD_KEYS,
    **CLINICAL_FIELD_KEYS,
}

A43_SURFACE_LABELS = {
    "mesial": "近中面",
    "distal": "远中面",
    "occlusal": "咬合面",
    "buccal": "颊／唇面",
    "labial": "颊／唇面",
    "lingual": "舌／腭面",
    "palatal": "舌／腭面",
}
A43_SURFACE_ALIASES = {
    **A43_SURFACE_LABELS,
    **{label: label for label in A43_SURFACE_LABELS.values()},
    "颊/唇面": "颊／唇面",
    "舌/腭面": "舌／腭面",
}
A43_GRADE_OPTIONS = ("无龋", "浅龋", "中龋", "深龋")

A41_GRADE_LABELS = tuple(f"第{i}阶段" for i in range(1, 7))
A41_GRADE_ALIASES = {label: f"S{i}" for i, label in enumerate(A41_GRADE_LABELS, 1)}
A41_GRADE_ALIASES.update({f"S{i}": f"S{i}" for i in range(1, 7)})
A41_GRADE_ALIASES.update({
    label: f"S{i}"
    for i, label in enumerate(("第一阶段", "第二阶段", "第三阶段", "第四阶段", "第五阶段", "第六阶段"), 1)
})

MODALITY_OPTIONS = ("口内照", "黏膜照", "根尖片", "全景片", "头颅侧位片")
A11_VIEW_OPTIONS_BY_MODALITY = {
    "口内照": (
        "全口", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
        "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
    ),
    "黏膜照": ("舌背", "舌腹", "舌缘", "左颊黏膜", "右颊黏膜", "上唇", "下唇", "上牙弓", "下牙弓"),
    "根尖片": (
        "上颌", "下颌", "前牙区", "右侧后牙区", "左侧后牙区",
        "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
    ),
    "全景片": ("全口",),
    "头颅侧位片": ("头颅侧位",),
}
VIEW_OPTIONS_BY_MODALITY = {
    "口内照": (
        "全口", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
        "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
    ),
    "黏膜照": (
        "舌背", "舌腹", "舌缘", "舌根", "舌尖", "左颊黏膜", "右颊黏膜", "上唇", "下唇",
        "腭部", "口底", "上牙龈", "下牙龈", "上牙弓", "下牙弓", "其他口腔部位",
    ),
    "根尖片": (
        "上颌", "下颌", "前牙区", "右侧后牙区", "左侧后牙区",
        "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
    ),
    "全景片": ("全口",),
    "头颅侧位片": ("头颅侧位",),
}
NON_MUCOSA_VIEW_OPTIONS = (
    "全口", "头颅侧位", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
    "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
)
A12_NON_MUCOSA_COVERAGE_OPTIONS = (
    "全口", "头颅侧位", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
    "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
)
A12_MUCOSA_COVERAGE_OPTIONS = (
    "舌背", "舌腹", "舌缘", "左颊黏膜", "右颊黏膜", "上唇", "下唇", "上牙弓", "下牙弓",
)
A12_COVERAGE_OPTIONS = tuple(
    dict.fromkeys((*A12_NON_MUCOSA_COVERAGE_OPTIONS, *A12_MUCOSA_COVERAGE_OPTIONS, "部位无法判断"))
)
A12_QUALITY_LABEL_TO_CODE = {
    "未见明显问题": "adequate",
    "覆盖或部位辨识问题": "coverage_or_site",
    "成像质量问题": "image_quality",
}
A12_QUALITY_CODE_TO_LABEL = {value: key for key, value in A12_QUALITY_LABEL_TO_CODE.items()}

SUFFICIENCY_OPTIONS = ("充分", "不足")
ACTION_OPTIONS = (
    "无需补充检查",
    "重拍同一区域",
    "补充缺失视角",
    "补充目标牙根尖片",
    "补充咬翼片",
    "按标准重拍全景片",
    "补充限视野CBCT",
    "先行针对性临床检查",
    "补充牙髓活力检查",
    "补充牙周探诊",
    "补充标准化口内照片",
    "补充侧位片",
    "补充多视角黏膜照片",
    "补充活检病理",
)
OLD_ACTION_CODE_TO_LABEL = {
    "NO_ADDITIONAL_ACTION": "无需补充检查",
    "RETAKE_SAME_REGION": "重拍同一区域",
    "ADD_MISSING_VIEW": "补充缺失视角",
    "ADD_TARGETED_PERIAPICAL": "补充目标牙根尖片",
    "ADD_BITEWING": "补充咬翼片",
    "ADD_LIMITED_FOV_CBCT": "补充限视野CBCT",
    "CLINICAL_EXAM_FIRST": "先行针对性临床检查",
    "PULP_VITALITY_TEST": "补充牙髓活力检查",
    "ADD_PERIODONTAL_PROBING": "补充牙周探诊",
    "PERIODONTAL_PROBING": "补充牙周探诊",
    "ADD_STANDARDIZED_ORAL_PHOTOS": "补充标准化口内照片",
    "ADD_MULTI_VIEW_MUCOSA_PHOTOS": "补充多视角黏膜照片",
    "BIOPSY_PATHOLOGY": "补充活检病理",
    "REPEAT_STANDARD_PANORAMIC": "按标准重拍全景片",
    "OTHER_REVIEW": "其他 / 待审核",
}
DEPARTMENT_LABELS = {
    "ENDODONTICS": "牙体牙髓科",
    "ORAL_SURGERY": "口腔颌面外科",
    "PERIODONTOLOGY": "牙周科",
    "ORAL_MEDICINE": "口腔黏膜科",
    "PROSTHODONTICS": "口腔修复科",
    "PEDIATRIC_DENTISTRY": "儿童口腔科",
    "IMPLANTOLOGY": "口腔种植科",
    "ORTHODONTICS": "口腔正畸科",
}
GOAL_LABELS = {
    "ENDO_CARIES_ASSESSMENT": "龋损或继发龋评估",
    "ENDO_PERIAPICAL_ASSESSMENT": "根尖周异常评估",
    "ENDO_ROOT_CANAL_TREATMENT": "根管治疗及根管填充情况评估",
    "IMPLANT_PREOP_BONE": "种植术前骨量初步评估",
    "MUCOSA_MORPHOLOGY": "黏膜病变可见形态评估",
    "ORTHO_RECORD_COMPLETENESS": "正畸治疗前资料完整性判断",
    "PED_TOOTH_GERM_ERUPTION": "儿童牙胚/萌出评估",
    "PERIO_BONE_LOSS": "牙周骨丧失评估",
    "PERIO_STAGING_SUFFICIENCY": "牙周分期所需资料充分性判断",
    "PROS_PLAN_SUFFICIENCY": "修复方案制定所需资料充分性判断",
    "SURGERY_ANATOMIC_RELATIONSHIP": "外科解剖关系评估",
    "SURGERY_IMPACTED_POSITION": "阻生牙位置评估",
}
DENTITION_STAGE_OPTIONS = ("乳牙列", "混合牙列", "恒牙列")
UNEVALUABLE_REASON_OPTIONS = ("牙列覆盖不足", "年龄信息不足", "图像质量不足")
TOOTH_DEVELOPMENT_STATUS_OPTIONS = ("已萌出", "萌出中", "未萌出", "阻生", "缺失")
TOOTH_DEVELOPMENT_STATUS_ALIASES = {
    "缺牙": "缺失",
}
VALID_FDI_CODES = {
    *range(11, 19),
    *range(21, 29),
    *range(31, 39),
    *range(41, 49),
    *range(51, 56),
    *range(61, 66),
    *range(71, 76),
    *range(81, 86),
}
FDI_LEFT_TO_RIGHT_ORDER = (
    18, 17, 16, 15, 14, 13, 12, 11,
    21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41,
    31, 32, 33, 34, 35, 36, 37, 38,
    55, 54, 53, 52, 51,
    61, 62, 63, 64, 65,
    85, 84, 83, 82, 81,
    71, 72, 73, 74, 75,
)
FDI_LEFT_TO_RIGHT_RANK = {fdi: rank for rank, fdi in enumerate(FDI_LEFT_TO_RIGHT_ORDER)}
A31_FINDING_CODES = (
    "龋样缺损", "非龋性牙体缺损", "牙体折裂", "根尖周透射影", "根尖周阻射影",
    "牙周骨吸收", "牙根吸收", "牙根形态异常", "骨质异常",
)
A31_REGION_CODES = (
    "全口", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
    "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
)
A32_FINDING_CODES = (
    "白色改变", "红色改变", "红白混合改变", "色素性改变", "溃疡", "糜烂",
    "水疱 / 大疱", "斑块样隆起", "丘疹 / 结节", "肿物", "萎缩", "其他/待审核",
)
A32_PROMPT_FINDING_CODES = A32_FINDING_CODES[:-1]
A32_FINDING_CODE_ALIASES = {
    "水疱/大疱": "水疱 / 大疱",
    "丘疹/结节": "丘疹 / 结节",
}
A32_REGION_CODES = (
    "舌背", "舌腹", "舌缘", "左颊黏膜", "右颊黏膜", "上唇", "下唇", "腭部", "上牙弓", "下牙弓",
)
A32_MODE_ALIASES = {
    "site_only": "region", "region": "region", "区域定位": "region",
    "box": "bbox", "bbox": "bbox", "画框定位": "bbox",
}
A33_FINDING_CODES = (
    "阻生", "异位萌出", "萌出延迟", "多生牙", "先天缺牙", "错位牙", "牙齿旋转",
    "拥挤", "牙列间隙", "深覆合", "深覆盖", "反合", "交叉合", "开合", "其他/待审核",
)
A33_PROMPT_FINDING_CODES = A33_FINDING_CODES[:-1]
A33_REGION_CODES = (
    "全口", "头颅侧位", "上颌", "下颌", "正面", "前牙区", "右侧后牙区", "左侧后牙区",
    "上颌前牙区", "下颌前牙区", "右上后牙区", "左上后牙区", "右下后牙区", "左下后牙区",
)
A33_FINDING_CODE_ALIASES = {}
A33_MODE_ALIASES = {
    "site_only": "region", "region": "region", "区域定位": "region",
    "tooth": "tooth", "fdi": "tooth", "牙位定位": "tooth",
    "box": "bbox", "bbox": "bbox", "画框定位": "bbox",
}
A34_FINDING_CODES = (
    "充填体", "嵌体 / 高嵌体", "根管充填", "桩核", "牙冠", "联冠", "固定桥",
    "种植体", "种植上部修复体", "正畸装置", "残根", "其他/待审核",
)
A34_PROMPT_FINDING_CODES = A34_FINDING_CODES[:-1]
A34_REGION_CODES = A33_REGION_CODES
A34_FINDING_CODE_ALIASES = {
    "嵌体/高嵌体": "嵌体 / 高嵌体",
}
A34_MODE_ALIASES = {
    "site_only": "region", "region": "region", "区域定位": "region",
    "tooth": "tooth", "fdi": "tooth", "牙位定位": "tooth",
    "box": "bbox", "bbox": "bbox", "画框定位": "bbox",
}
A31_MODE_ALIASES = {
    "local": "tooth", "fdi": "tooth", "tooth": "tooth", "site_only": "tooth", "牙位定位": "tooth",
    "box": "bbox", "bbox": "bbox", "画框定位": "bbox",
    "region": "region", "区域定位": "region",
}
CEPHALOMETRIC_STRUCTURE_LABELS = {
    "A": "A-point",
    "ANS": "Anterior Nasal Spine",
    "B": "B-point",
    "Me": "Menton",
    "N": "Nasion",
    "Or": "Orbitale",
    "Pog": "Pogonion",
    "PNS": "Posterior Nasal Spine",
    "Pn": "Pronasale",
    "R": "Ramus",
    "S": "Sella",
    "Ar": "Articulare",
    "Co": "Condylion",
    "Gn": "Gnathion",
    "Go": "Gonion",
    "Po": "Porion",
    "LPM": "Lower 2nd PM Cusp Tip",
    "LIT": "Lower Incisor Tip",
    "LMT": "Lower Molar Cusp Tip",
    "UPM": "Upper 2nd PM Cusp Tip",
    "UIA": "Upper Incisor Apex",
    "UIT": "Upper Incisor Tip",
    "UMT": "Upper Molar Cusp Tip",
    "LIA": "Lower Incisor Apex",
    "Li": "Labrale inferius",
    "Ls": "Labrale superius",
    "N`": "Soft Tissue Nasion",
    "Pog`": "Soft Tissue Pogonion",
    "Sn": "Subnasale",
}
CEPHALOMETRIC_STRUCTURE_DESCRIPTIONS = {
    "S": "S（Sella，蝶鞍点）：蝶鞍（垂体窝）影像的中心点。",
    "N": "N（Nasion，鼻根点）：鼻额缝，即鼻骨与额骨交界处的最前点。",
    "Po": "Po（Porion，耳点）：骨性外耳道上缘的最高点，即解剖耳点。",
    "Or": "Or（Orbitale，眶点）：骨性眼眶下缘的最低点。",
    "ANS": "ANS（Anterior Nasal Spine，前鼻棘点）：前鼻棘的尖端。",
    "PNS": "PNS（Posterior Nasal Spine，后鼻棘点）：硬腭后方骨性鼻棘的尖端。",
    "A": "A（A-point / Subspinale，上牙槽座点）：前鼻棘与上齿槽缘之间，上颌骨前缘轮廓的最凹点。",
    "R": "R（Ramus，下颌支点）：下颌支后方外轮廓沿垂直段的最凸点。",
    "Ar": "Ar（Articulare，关节点）：通常指颅底下缘与下颌支或髁突颈后缘影像的交点。",
    "Co": "Co（Condylion，髁突点／髁顶点）：下颌髁突轮廓的最后上点。",
    "Gn": "Gn（Gnathion，颏顶点）：骨性颏部前下方的定位点，通常在颏前点与颏下点之间的轮廓上按规定方法确定。",
    "Go": "Go（Gonion，下颌角点）：下颌支后缘与下颌体下缘之间，下颌角轮廓的中间定位点。",
    "B": "B（B-point / Supramentale，下牙槽座点）：下齿槽缘与颏前点之间，下颌联合部前缘轮廓的最凹点。",
    "Pog": "Pog（Pogonion，颏前点）：下颌联合部骨性颏部轮廓的最前突点。",
    "Me": "Me（Menton，颏下点）：下颌联合部骨性轮廓的最低点。",
    "LPM": "LPM（Lower 2nd Premolar Cusp Tip，下颌第二前磨牙颊尖点）：下颌第二前磨牙颊侧牙尖的尖端。",
    "LIT": "LIT（Lower Incisor Tip，下中切牙切缘点）：下中切牙的切缘定位点。",
    "LMT": "LMT（Lower Molar Cusp Tip，下颌第一磨牙近中颊尖点）：下颌第一磨牙近中颊侧牙尖的尖端。",
    "UPM": "UPM（Upper 2nd Premolar Cusp Tip，上颌第二前磨牙颊尖点）：上颌第二前磨牙颊侧牙尖的尖端。",
    "UIA": "UIA（Upper Incisor Apex，上中切牙根尖点）：上中切牙牙根的根尖。",
    "UIT": "UIT（Upper Incisor Tip，上中切牙切缘点）：上中切牙的切缘定位点。",
    "UMT": "UMT（Upper Molar Cusp Tip，上颌第一磨牙近中颊尖点）：上颌第一磨牙近中颊侧牙尖的尖端。",
    "LIA": "LIA（Lower Incisor Apex，下中切牙根尖点）：下中切牙牙根的根尖。",
    "Pn": "Pn（Pronasale，鼻尖点／鼻顶点）：鼻尖软组织轮廓的最前突点。",
    "Li": "Li（Labrale inferius，下唇点）：正中矢状面上，下唇唇红缘的最突出点。",
    "Ls": "Ls（Labrale superius，上唇点）：正中矢状面上，上唇唇红缘的最突出点。",
    "N`": "N`（Soft Tissue Nasion，软组织鼻根点）：骨性鼻根点附近、鼻根表面软组织轮廓上的对应定位点。",
    "Pog`": "Pog`（Soft Tissue Pogonion，软组织颏前点）：软组织颏部轮廓的最前突点。",
    "Sn": "Sn（Subnasale，鼻下点）：正中矢状面上，鼻小柱基底与上唇相接的点。",
}
def describe_a24_cephalometric_structure(code: str) -> str:
    return CEPHALOMETRIC_STRUCTURE_DESCRIPTIONS.get(code, code)
CEPHALOMETRIC_STRUCTURE_ALIASES = {
    **{label: code for code, label in CEPHALOMETRIC_STRUCTURE_LABELS.items()},
    "A点": "A",
    "A point": "A",
    "鼻根点": "N",
    "蝶鞍点": "S",
    "眶点": "Or",
    "耳点": "Po",
    "前鼻棘": "ANS",
    "后鼻棘": "PNS",
    "颏下点": "Me",
    "颏前点": "Pog",
    "软组织鼻根点": "N`",
    "软组织颏前点": "Pog`",
    "N′": "N`",
    "Pog′": "Pog`",
    "鼻下点": "Sn",
    "鼻尖点": "Pn",
}
PERIAPICAL_STRUCTURE_CODE_TO_LABEL = {
    "MESIAL_CEJ": "近中釉牙骨质界",
    "DISTAL_CEJ": "远中釉牙骨质界",
    "ROOT_APEX": "根尖",
    "MESIAL_ALVEOLAR_CREST": "近中牙槽嵴顶",
    "DISTAL_ALVEOLAR_CREST": "远中牙槽嵴顶",
    "FURCATION": "根分叉",
}
PERIAPICAL_STRUCTURE_LABEL_TO_CODE = {
    label: code for code, label in PERIAPICAL_STRUCTURE_CODE_TO_LABEL.items()
}


def fields_to_dict(fields: list[dict[str, Any]], task_code: str | None = None) -> dict[str, Any]:
    values = {}
    for item in fields:
        key = str(item["key"])
        value = item["value"]
        if key == "minimum_necessary_action" and isinstance(value, dict):
            value = normalize_action_value(value)
        if key == "coverage_codes":
            value = normalize_coverage_codes(value)
        elif key == "quality_grade":
            value = normalize_quality_grade(value)
        elif key == "teeth":
            value = normalize_a22_teeth(value)
        elif key == "structures":
            value = normalize_a24_structures(value)
        elif key == "findings":
            if task_code == "A3.2":
                value = normalize_a32_findings(value)
            elif task_code == "A3.3":
                value = normalize_a33_findings(value)
            elif task_code == "A3.4":
                value = normalize_a34_findings(value)
            else:
                value = normalize_a31_findings(value)
        elif key == "surface_grades":
            value = normalize_a43_surface_grades(value)
        elif key == "target_teeth":
            key = "teeth"
            value = normalize_a23_teeth(value)
        elif key == "unevaluable_reasons":
            value = normalize_unevaluable_reasons(value)
        elif not isinstance(value, list):
            value = str(value).strip()
        values[key] = value
    return values


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def format_options(options: tuple[str, ...]) -> str:
    return " ".join(f"{option_letter(i)}. {value}" for i, value in enumerate(options))


def option_to_value(letter: str, options: tuple[str, ...]) -> str:
    normalized = str(letter).strip().upper()
    if len(normalized) != 1 or not normalized.isalpha():
        raise ValueError(f"invalid option letter: {letter}")

    index = ord(normalized) - ord("A")
    if index < 0 or index >= len(options):
        raise ValueError(f"option {letter} out of range for options: {options}")
    return options[index]


def get_a31_scope(case: dict[str, Any]) -> dict[str, Any]:
    raw = get_field_value(case, "evaluation_scope")
    if not isinstance(raw, dict):
        raw = {}
    concepts = raw.get("concept_codes") or []
    concepts = [str(item).strip() for item in concepts if str(item).strip() and str(item) != "__UNSPECIFIED_FINDING__"]
    site = raw.get("site") or {}
    site_type = str(site.get("type", "whole_mouth")).strip()
    fdis = [normalize_fdi(item) for item in site.get("fdi", [])] if site_type == "tooth" else []
    regions = [str(item).strip() for item in site.get("region_codes", []) if str(item).strip()]
    return {"concepts": concepts, "site_type": site_type, "fdis": sorted(set(fdis)), "regions": regions}


def get_a31_mode(case: dict[str, Any]) -> str:
    value = get_field_value(case, "evaluation_scope")
    mode = value.get("localization_mode") if isinstance(value, dict) else None
    fallback = A31_MODE_ALIASES.get(str(mode or "local").strip(), str(mode or "local").strip())
    return infer_reference_localization_mode(case, A31_MODE_ALIASES, fallback)


def _a31_reference_finding_codes(case: dict[str, Any]) -> list[str]:
    """Return unique reference labels used to build the case-local choices."""
    for field in case.get("reference_answer", {}).get("fields", []):
        if field.get("key") != "findings" or not isinstance(field.get("value"), list):
            continue
        codes = {
            str(item.get("finding_code")).strip()
            for item in field["value"]
            if isinstance(item, dict) and str(item.get("finding_code", "")).strip() in A31_FINDING_CODES
        }
        return [code for code in A31_FINDING_CODES if code in codes]
    return []


def _case_stable_rng(case: dict[str, Any]) -> random.Random:
    """Create the shared reproducible per-case RNG for randomized selections."""
    seed_text = "{}|{}|{}".format(
        case.get("task_code", ""),
        case.get("id", ""),
        case.get("case_code", ""),
    )
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def select_a31_prompt_options(case: dict[str, Any]) -> list[str]:
    """Build six deterministic choices containing every correct A3.1 label."""
    correct = _a31_reference_finding_codes(case)
    if len(correct) > 6:
        raise ValueError(f"A3.1 has more than six correct finding codes: {correct}")
    pool = A31_FINDING_CODES
    distractors = [code for code in pool if code not in correct]
    rng = _case_stable_rng(case)
    rng.shuffle(distractors)
    options = [*correct, *distractors[: 6 - len(correct)]]
    rng.shuffle(options)
    return options


def infer_reference_localization_mode(case: dict[str, Any], aliases: dict[str, str], fallback: str) -> str:
    findings: list[dict[str, Any]] = []
    for field in case.get("reference_answer", {}).get("fields", []):
        if field.get("key") == "findings" and isinstance(field.get("value"), list):
            findings = field["value"]
            break
    if any("bbox" in item or "normalized_bbox" in item for item in findings if isinstance(item, dict)):
        return aliases.get("bbox", "bbox")
    if any(
        "fdi" in item
        or (isinstance(item.get("site"), dict) and item["site"].get("type") == "tooth")
        for item in findings
        if isinstance(item, dict)
    ):
        return aliases.get("tooth", "tooth")
    if any(
        "region_code" in item
        or "region_codes" in item
        or (isinstance(item.get("site"), dict) and item["site"].get("type") == "region")
        for item in findings
        if isinstance(item, dict)
    ):
        return aliases.get("region", "region")
    return fallback


def get_a32_scope(case: dict[str, Any]) -> dict[str, Any]:
    raw = get_field_value(case, "evaluation_scope")
    if not isinstance(raw, dict):
        raw = {}
    other_text = str(raw.get("other_text") or "").strip()
    concepts = [
        str(item).strip()
        for item in raw.get("concept_codes", []) or []
        if str(item).strip() and str(item) != "__UNSPECIFIED_FINDING__"
    ]
    concepts = [concept for concept in concepts if concept != "其他/待审核"]
    if other_text and "其他/待审核" in (raw.get("concept_codes") or []):
        concepts.append(other_text)
    site = raw.get("site") or {}
    regions = [str(item).strip() for item in site.get("region_codes", []) if str(item).strip()]
    fallback = A32_MODE_ALIASES.get(str(raw.get("localization_mode") or "region").strip(), "region")
    mode = infer_reference_localization_mode(case, A32_MODE_ALIASES, fallback)
    return {"concepts": concepts, "other_text": other_text, "regions": regions, "mode": mode}


def _a32_reference_finding_codes(case: dict[str, Any]) -> list[str]:
    """Return unique reference labels used to build the six A3.2 choices."""
    for field in case.get("reference_answer", {}).get("fields", []):
        if field.get("key") != "findings" or not isinstance(field.get("value"), list):
            continue
        codes: list[str] = []
        for item in field["value"]:
            if not isinstance(item, dict):
                continue
            raw_code = str(item.get("finding_code", "")).strip()
            code = A32_FINDING_CODE_ALIASES.get(raw_code, raw_code)
            if code == "其他/待审核":
                code = str(item.get("other_text") or code).strip()
            if code and code not in codes:
                codes.append(code)
        return codes
    return []


def select_a32_prompt_options(case: dict[str, Any]) -> list[str]:
    """Build six deterministic choices containing every correct A3.2 label."""
    correct = _a32_reference_finding_codes(case)
    if len(correct) > 6:
        raise ValueError(f"A3.2 has more than six correct finding codes: {correct}")
    pool = list(A32_PROMPT_FINDING_CODES)
    distractors = [code for code in pool if code not in correct]
    rng = _case_stable_rng(case)
    rng.shuffle(distractors)
    options = [*correct, *distractors[: 6 - len(correct)]]
    rng.shuffle(options)
    return options


def get_a33_scope(case: dict[str, Any]) -> dict[str, Any]:
    raw = get_field_value(case, "evaluation_scope")
    if not isinstance(raw, dict):
        raw = {}
    raw_concepts = [str(item).strip() for item in raw.get("concept_codes", []) or [] if str(item).strip()]
    other_text = str(raw.get("other_text") or "").strip()
    concepts = [concept for concept in raw_concepts if concept not in {"__UNSPECIFIED_FINDING__", "其他/待审核"}]
    if other_text and "其他/待审核" in raw_concepts:
        concepts.append(other_text)
    site = raw.get("site") or {}
    site_type = str(site.get("type", "whole_mouth")).strip()
    fdis = [normalize_fdi(item) for item in site.get("fdi", [])] if site_type == "tooth" else []
    regions = [str(item).strip() for item in site.get("region_codes", []) if str(item).strip()]
    raw_mode = raw.get("localization_mode")
    if raw_mode:
        mode = A33_MODE_ALIASES.get(str(raw_mode).strip(), str(raw_mode).strip())
    elif site_type == "tooth":
        mode = "tooth"
    else:
        mode = "region"
    mode = infer_reference_localization_mode(case, A33_MODE_ALIASES, mode)
    return {"concepts": concepts, "other_text": other_text, "site_type": site_type, "fdis": sorted(set(fdis)), "regions": regions, "mode": mode}


def _a33_reference_finding_codes(case: dict[str, Any]) -> list[str]:
    """Return unique reference labels used to build the six A3.3 choices."""
    for field in case.get("reference_answer", {}).get("fields", []):
        if field.get("key") != "findings" or not isinstance(field.get("value"), list):
            continue
        codes: list[str] = []
        for item in field["value"]:
            if not isinstance(item, dict):
                continue
            code = str(item.get("finding_code", "")).strip()
            if code == "其他/待审核":
                code = str(item.get("other_text") or code).strip()
            if code and code not in codes:
                codes.append(code)
        return codes
    return []


def select_a33_prompt_options(case: dict[str, Any]) -> list[str]:
    """Build six deterministic choices containing every correct A3.3 label."""
    correct = _a33_reference_finding_codes(case)
    if len(correct) > 6:
        raise ValueError(f"A3.3 has more than six correct finding codes: {correct}")
    pool = list(A33_PROMPT_FINDING_CODES)
    distractors = [code for code in pool if code not in correct]
    rng = _case_stable_rng(case)
    rng.shuffle(distractors)
    options = [*correct, *distractors[: 6 - len(correct)]]
    rng.shuffle(options)
    return options


def get_a34_scope(case: dict[str, Any]) -> dict[str, Any]:
    raw = get_field_value(case, "evaluation_scope")
    if not isinstance(raw, dict):
        raw = {}
    raw_concepts = [str(item).strip() for item in raw.get("concept_codes", []) or [] if str(item).strip()]
    other_text = str(raw.get("other_text") or "").strip()
    concepts = [concept for concept in raw_concepts if concept not in {"__UNSPECIFIED_FINDING__", "其他/待审核"}]
    if other_text and "其他/待审核" in raw_concepts:
        concepts.append(other_text)
    site = raw.get("site") or {}
    site_type = str(site.get("type", "whole_mouth")).strip()
    fdis = [normalize_fdi(item) for item in site.get("fdi", [])] if site_type == "tooth" else []
    regions = [str(item).strip() for item in site.get("region_codes", []) if str(item).strip()]
    raw_mode = raw.get("localization_mode")
    if raw_mode:
        mode = A34_MODE_ALIASES.get(str(raw_mode).strip(), str(raw_mode).strip())
    elif site_type == "tooth":
        mode = "tooth"
    else:
        mode = "region"
    mode = infer_reference_localization_mode(case, A34_MODE_ALIASES, mode)
    return {"concepts": concepts, "site_type": site_type, "fdis": sorted(set(fdis)), "regions": regions, "mode": mode}


def _a34_reference_finding_codes(case: dict[str, Any]) -> list[str]:
    """Return unique reference labels used to build the six A3.4 choices."""
    for field in case.get("reference_answer", {}).get("fields", []):
        if field.get("key") != "findings" or not isinstance(field.get("value"), list):
            continue
        codes: list[str] = []
        for item in field["value"]:
            if not isinstance(item, dict):
                continue
            raw_code = str(item.get("finding_code", "")).strip()
            code = A34_FINDING_CODE_ALIASES.get(raw_code, raw_code)
            if code == "其他/待审核":
                code = str(item.get("other_text") or code).strip()
            if code and code not in codes:
                codes.append(code)
        return codes
    return []


def select_a34_prompt_options(case: dict[str, Any]) -> list[str]:
    """Build six deterministic choices containing every correct A3.4 label."""
    correct = _a34_reference_finding_codes(case)
    if len(correct) > 6:
        raise ValueError(f"A3.4 has more than six correct finding codes: {correct}")
    distractors = [code for code in A34_PROMPT_FINDING_CODES if code not in correct]
    rng = _case_stable_rng(case)
    rng.shuffle(distractors)
    options = [*correct, *distractors[: 6 - len(correct)]]
    rng.shuffle(options)
    return options


def get_a43_target_surfaces(case: dict[str, Any]) -> list[str]:
    for field in case.get("input", {}).get("fields", []):
        if field.get("key") == "target_surfaces":
            raw_surfaces = field.get("value", [])
            if not isinstance(raw_surfaces, list):
                raise ValueError("target_surfaces must be a list")
            return [A43_SURFACE_LABELS.get(str(surface).strip(), str(surface).strip()) for surface in raw_surfaces]
    raise ValueError(f"case has no target_surfaces: {case.get('id')}")


def parse_json_object(text: str | None) -> dict[str, Any]:
    if text is None:
        raise ValueError("empty model response")

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")

    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates.extend(fenced)

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first:last + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(obj, dict):
            last_error = ValueError("parsed JSON is not an object")
            continue
        return {str(key): value for key, value in obj.items()}

    raise ValueError(f"cannot parse JSON object from response: {last_error}")


def parse_a11_answer(text: str | None) -> dict[str, str]:
    obj = parse_json_object(text)
    missing = [key for key in A11_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A11_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")

    parsed = {
        "modality_type": str(obj["modality_type"]).strip(),
        "field_of_view": str(obj["field_of_view"]).strip(),
    }
    if parsed["modality_type"] not in MODALITY_OPTIONS:
        raise ValueError(f"invalid modality_type: {parsed['modality_type']}")
    all_view_options = {
        view
        for options in A11_VIEW_OPTIONS_BY_MODALITY.values()
        for view in options
    }
    if parsed["field_of_view"] not in all_view_options:
        raise ValueError(f"invalid field_of_view: {parsed['field_of_view']}")
    return parsed


def parse_a12_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A12_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A12_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")

    return {
        "coverage_codes": normalize_coverage_codes(obj["coverage_codes"]),
        "quality_grade": normalize_quality_grade(obj["quality_grade"]),
    }


def parse_a13_answer(text: str | None) -> dict[str, str]:
    obj = parse_json_object(text)
    missing = [key for key in A13_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A13_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")

    parsed = {
        "sufficiency": str(obj["sufficiency"]).strip(),
        "minimum_necessary_action": normalize_action_value(obj["minimum_necessary_action"]),
    }
    validate_a13_answer(parsed)
    return parsed


def parse_a21_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A21_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A21_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")

    parsed = {
        "dentition_stage": str(obj["dentition_stage"]).strip(),
    }
    validate_a21_answer(parsed)
    return parsed


def parse_a22_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A22_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A22_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"teeth": normalize_a22_teeth(obj["teeth"])}


def parse_a23_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A23_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A23_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"teeth": normalize_a23_teeth(obj["teeth"])}


def parse_a24_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A24_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A24_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"structures": normalize_a24_structures(obj["structures"])}


def parse_a31_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A31_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A31_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"findings": normalize_a31_findings(obj["findings"])}


def parse_a32_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A32_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A32_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"findings": normalize_a32_findings(obj["findings"])}


def parse_a33_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A33_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A33_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"findings": normalize_a33_findings(obj["findings"])}


def parse_a34_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A34_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A34_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    return {"findings": normalize_a34_findings(obj["findings"])}


def parse_a41_answer(text: str | None) -> dict[str, str]:
    obj = parse_json_object(text)
    missing = [key for key in A41_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A41_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    grade = str(obj["grade_value"]).strip()
    if grade not in A41_GRADE_ALIASES:
        raise ValueError(f"invalid grade_value: {grade}")
    return {"grade_value": A41_GRADE_ALIASES[grade]}


def parse_a42_answer(text: str | None) -> dict[str, str]:
    obj = parse_json_object(text)
    missing = [key for key in A42_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A42_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    grade = str(obj["grade_value"]).strip()
    if grade not in {"无吸收", "轻度吸收", "中度吸收", "重度吸收"}:
        raise ValueError(f"invalid grade_value: {grade}")
    return {"grade_value": grade}


def normalize_a43_surface_grades(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("surface_grades must be a list")
    normalized = []
    for item in value:
        if not isinstance(item, dict) or not {"target_surface", "grade_value"}.issubset(item):
            raise ValueError(f"invalid surface grade item: {item}")
        surface = A43_SURFACE_ALIASES.get(str(item["target_surface"]).strip())
        grade = str(item["grade_value"]).strip()
        if surface is None or grade not in A43_GRADE_OPTIONS:
            raise ValueError(f"invalid surface grade item: {item}")
        normalized.append({"target_surface": surface, "grade_value": grade})
    return normalized


def parse_a43_answer(text: str | None) -> dict[str, Any]:
    obj = parse_json_object(text)
    missing = [key for key in A43_ANSWER_KEYS if key not in obj]
    if missing:
        raise ValueError(f"missing required answer keys: {missing}")
    extra = [key for key in obj if key not in A43_ANSWER_KEYS]
    if extra:
        raise ValueError(f"unexpected answer keys: {extra}")
    raw_grades = obj["surface_grades"]
    if not isinstance(raw_grades, list):
        raise ValueError("surface_grades must be a list")
    for item in raw_grades:
        if not isinstance(item, dict) or set(item) != {"target_surface", "grade_value"}:
            raise ValueError(f"invalid surface grade item: {item}")
    return {"surface_grades": normalize_a43_surface_grades(obj["surface_grades"])}


def normalize_answer(value: str) -> str:
    text = re.sub(r"\s+", "", str(value)).strip()
    text = text.replace("粘", "黏")
    text = text.replace("９０", "90").replace("１８０", "180")
    text = text.replace("°", "").replace("度", "")
    return text


def validate_a11_answer(answer: dict[str, str]) -> None:
    modality = answer["modality_type"]
    if modality not in MODALITY_OPTIONS:
        raise ValueError(f"invalid modality_type: {modality}")

    if answer["field_of_view"] not in A11_VIEW_OPTIONS_BY_MODALITY[modality]:
        raise ValueError(
            f"field_of_view {answer['field_of_view']} is not compatible with modality_type {modality}"
        )


def normalize_coverage_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"coverage_codes must be a list: {value}")
    if not value:
        raise ValueError("coverage_codes must be non-empty")
    if len(value) != 1:
        raise ValueError("coverage_codes must contain exactly one item")

    coverage = []
    for item in value:
        code = str(item).strip()
        if code not in A12_COVERAGE_OPTIONS:
            raise ValueError(f"invalid coverage code: {code}")
        coverage.append(code)
    return coverage


def normalize_quality_grade(value: Any) -> str:
    text = str(value).strip()
    text = A12_QUALITY_LABEL_TO_CODE.get(text, text)
    if text not in A12_QUALITY_CODE_TO_LABEL:
        raise ValueError(f"invalid quality_grade: {value}")
    return text


def validate_a13_answer(answer: dict[str, str]) -> None:
    if answer["sufficiency"] not in SUFFICIENCY_OPTIONS:
        raise ValueError(f"invalid sufficiency: {answer['sufficiency']}")

    action_value = answer["minimum_necessary_action"]
    action_code, _, requested_view = action_value.partition(":")
    if action_code not in ACTION_OPTIONS:
        raise ValueError(f"invalid minimum_necessary_action: {action_value}")
    if action_code == "补充缺失视角":
        allowed_views = set(NON_MUCOSA_VIEW_OPTIONS) | set(VIEW_OPTIONS_BY_MODALITY["黏膜照"])
        if requested_view not in allowed_views:
            raise ValueError(f"invalid requested_view for 补充缺失视角: {requested_view}")


def validate_a21_answer(answer: dict[str, Any]) -> None:
    stage = answer["dentition_stage"]
    if stage not in DENTITION_STAGE_OPTIONS:
        raise ValueError(f"invalid dentition_stage: {stage}")


def normalize_unevaluable_reasons(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"unevaluable_reasons must be a list: {value}")

    reasons = []
    for item in value:
        reason = str(item).strip()
        if reason not in UNEVALUABLE_REASON_OPTIONS:
            raise ValueError(f"invalid unevaluable reason: {reason}")
        reasons.append(reason)
    return sorted(set(reasons), key=UNEVALUABLE_REASON_OPTIONS.index)


def normalize_a22_teeth(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"teeth must be a list: {value}")

    if not value:
        return []
    if all(isinstance(item, dict) for item in value):
        if all("bbox" not in item and "normalized_bbox" not in item for item in value):
            return sorted(set(normalize_fdi(item.get("fdi")) for item in value))
        if any("bbox" not in item and "normalized_bbox" not in item for item in value):
            raise ValueError("tooth bbox objects must include bbox or normalized_bbox for every item")
        teeth_by_fdi: dict[int, dict[str, Any]] = {}
        for item in value:
            fdi = normalize_fdi(item.get("fdi"))
            bbox_value = item.get("bbox", item.get("normalized_bbox"))
            teeth_by_fdi[fdi] = {
                "fdi": fdi,
                "bbox": normalize_bbox(bbox_value),
            }
        return [teeth_by_fdi[fdi] for fdi in sorted(teeth_by_fdi)]
    if all(not isinstance(item, dict) for item in value):
        return sorted(set(normalize_fdi(item) for item in value))
    raise ValueError("teeth must contain either only FDI integers or only tooth bbox objects")


def normalize_a23_teeth(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"teeth must be a list: {value}")

    teeth_by_fdi: dict[int, dict[str, Any]] = {}
    fdi_order: list[int] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"tooth status item must be an object: {item}")
        if "fdi" not in item or "status" not in item:
            raise ValueError(f"tooth status item must include fdi and status: {item}")

        fdi = normalize_fdi(item.get("fdi"))
        status = normalize_tooth_development_status(item.get("status"))
        if status not in TOOTH_DEVELOPMENT_STATUS_OPTIONS:
            raise ValueError(f"invalid tooth status: {status}")
        if fdi not in teeth_by_fdi:
            fdi_order.append(fdi)
        teeth_by_fdi[fdi] = {"fdi": fdi, "status": status}

    return [teeth_by_fdi[fdi] for fdi in fdi_order]


def normalize_a24_structures(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"structures must be a list: {value}")

    structures: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"structure item must be an object: {item}")
        if "structure_code" not in item or "coordinate" not in item:
            raise ValueError(f"structure item must include structure_code and coordinate: {item}")

        structure_code = normalize_a24_structure_code(item["structure_code"])
        coordinate = normalize_coordinate(item["coordinate"])
        fdi = normalize_fdi(item["fdi"]) if "fdi" in item and item["fdi"] not in (None, "") else None
        key = (structure_code, fdi)
        if key in seen:
            continue
        seen.add(key)

        normalized = {
            "structure_code": structure_code,
            "coordinate": coordinate,
        }
        if fdi is not None:
            normalized["fdi"] = fdi
        structures.append(normalized)
    return structures


def normalize_a31_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"findings must be a list: {value}")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or "finding_code" not in item:
            raise ValueError(f"finding item must include finding_code: {item}")
        code = str(item["finding_code"]).strip()
        if code not in A31_FINDING_CODES:
            raise ValueError(f"invalid finding_code: {code}")
        site = item.get("site") if isinstance(item.get("site"), dict) else {}
        if "fdi" in item or "fdi" in site:
            fdi_values = item.get("fdi", site.get("fdi", []))
            fdis = sorted(set(normalize_fdi(fdi) for fdi in fdi_values))
            if not fdis:
                raise ValueError("fdi must be non-empty")
            normalized.append({"finding_code": code, "fdi": fdis})
        elif "bbox" in item or "normalized_bbox" in item:
            normalized.append({"finding_code": code, "bbox": normalize_bbox(item.get("bbox", item.get("normalized_bbox")))})
        else:
            region = item.get("region_code", item.get("region", site))
            if isinstance(region, dict):
                regions = region.get("region_codes", [])
            else:
                regions = region if isinstance(region, list) else [region]
            for region_code in regions:
                region_code = str(region_code).strip()
                if region_code not in A31_REGION_CODES:
                    raise ValueError(f"invalid region_code: {region_code}")
                normalized.append({"finding_code": code, "region_code": region_code})
    return normalized


def normalize_a32_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"findings must be a list: {value}")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or "finding_code" not in item:
            raise ValueError(f"finding item must include finding_code: {item}")
        raw_code = str(item["finding_code"]).strip()
        code = A32_FINDING_CODE_ALIASES.get(raw_code, raw_code)
        if code == "其他/待审核":
            code = str(item.get("other_text") or "其他/待审核").strip()
        if not code:
            raise ValueError("finding_code must be non-empty")
        if "bbox" in item or "normalized_bbox" in item:
            normalized.append({"finding_code": code, "bbox": normalize_bbox(item.get("bbox", item.get("normalized_bbox")))})
            continue
        site = item.get("site") if isinstance(item.get("site"), dict) else {}
        regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
        if isinstance(regions, str):
            regions = [regions]
        if not isinstance(regions, list) or not regions:
            raise ValueError("region_code must be a non-empty list")
        cleaned = [str(region).strip() for region in regions]
        if any(region not in A32_REGION_CODES for region in cleaned):
            raise ValueError(f"invalid soft-tissue region_code: {cleaned}")
        normalized.append({
            "finding_code": code,
            "region_code": list(dict.fromkeys(cleaned)),
        })
    return normalized


def normalize_a33_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"findings must be a list: {value}")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or "finding_code" not in item:
            raise ValueError(f"finding item must include finding_code: {item}")
        raw_code = str(item["finding_code"]).strip()
        code = A33_FINDING_CODE_ALIASES.get(raw_code, raw_code)
        if code == "其他/待审核":
            code = str(item.get("other_text") or "其他/待审核").strip()
        if not code:
            raise ValueError("finding_code must be non-empty")
        if "bbox" in item or "normalized_bbox" in item:
            normalized.append({"finding_code": code, "bbox": normalize_bbox(item.get("bbox", item.get("normalized_bbox")))})
        elif "fdi" in item:
            fdis = sorted(set(normalize_fdi(fdi) for fdi in item["fdi"]))
            if not fdis:
                raise ValueError("fdi must be non-empty")
            normalized.append({"finding_code": code, "fdi": fdis})
        else:
            site = item.get("site") if isinstance(item.get("site"), dict) else {}
            if site.get("type") == "tooth":
                fdis = sorted(set(normalize_fdi(fdi) for fdi in site.get("fdi", [])))
                if not fdis:
                    raise ValueError("site.fdi must be non-empty")
                normalized.append({"finding_code": code, "fdi": fdis})
                continue
            regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
            if isinstance(regions, str):
                regions = [regions]
            if not isinstance(regions, list) or not regions:
                raise ValueError("region_code must be a non-empty list")
            cleaned = [str(region).strip() for region in regions]
            if any(region not in A33_REGION_CODES for region in cleaned):
                raise ValueError(f"invalid orthodontic region_code: {cleaned}")
            normalized.append({"finding_code": code, "region_code": list(dict.fromkeys(cleaned))})
    return normalized


def normalize_a34_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"findings must be a list: {value}")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or "finding_code" not in item:
            raise ValueError(f"finding item must include finding_code: {item}")
        raw_code = str(item["finding_code"]).strip()
        code = A34_FINDING_CODE_ALIASES.get(raw_code, raw_code)
        if code == "其他/待审核":
            code = str(item.get("other_text") or "其他/待审核").strip()
        if not code:
            raise ValueError("finding_code must be non-empty")
        if "bbox" in item or "normalized_bbox" in item:
            normalized.append({"finding_code": code, "bbox": normalize_bbox(item.get("bbox", item.get("normalized_bbox")))})
            continue
        if "fdi" in item:
            fdis = _normalize_a34_tooth_sequence(item["fdi"], code)
            if not fdis:
                raise ValueError("fdi must be non-empty")
            normalized.append({"finding_code": code, "fdi": fdis})
            continue
        site = item.get("site") if isinstance(item.get("site"), dict) else {}
        if site.get("type") == "tooth":
            fdis = _normalize_a34_tooth_sequence(site.get("fdi", []), code)
            if not fdis:
                raise ValueError("site.fdi must be non-empty")
            normalized.append({"finding_code": code, "fdi": fdis})
            continue
        regions = item.get("region_code", item.get("region_codes", site.get("region_codes", [])))
        if isinstance(regions, str):
            regions = [regions]
        if not isinstance(regions, list) or not regions:
            raise ValueError("region_code must be a non-empty list")
        cleaned = [str(region).strip() for region in regions]
        if any(region not in A34_REGION_CODES for region in cleaned):
            raise ValueError(f"invalid treatment-status region_code: {cleaned}")
        normalized.append({"finding_code": code, "region_code": list(dict.fromkeys(cleaned))})
    return normalized


def _normalize_a34_tooth_sequence(values: Any, finding_code: str) -> list[int | str]:
    if not isinstance(values, list):
        raise ValueError(f"fdi must be a list: {values}")
    normalized: list[int | str] = []
    for value in values:
        if isinstance(value, str) and value.strip().lower() == "x":
            if finding_code != "固定桥":
                raise ValueError("x is only valid for fixed-bridge pontic positions")
            normalized.append("x")
        else:
            normalized.append(normalize_fdi(value))
    return list(dict.fromkeys(normalized))


def normalize_a24_structure_code(value: Any) -> str:
    code = str(value).strip()
    code = PERIAPICAL_STRUCTURE_LABEL_TO_CODE.get(code, code)
    return CEPHALOMETRIC_STRUCTURE_ALIASES.get(code, code)


def normalize_tooth_development_status(value: Any) -> str:
    status = str(value).strip()
    return TOOTH_DEVELOPMENT_STATUS_ALIASES.get(status, status)


def normalize_fdi(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid FDI value: {value}")
    try:
        fdi = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid FDI value: {value}") from exc
    if fdi not in VALID_FDI_CODES:
        raise ValueError(f"FDI value out of allowed range: {fdi}")
    return fdi


def normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"bbox must be a four-number list: {value}")
    try:
        left, top, right, bottom = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox values must be numbers: {value}") from exc
    if not (0 <= left <= 1 and 0 <= top <= 1 and 0 <= right <= 1 and 0 <= bottom <= 1):
        raise ValueError(f"bbox values must be normalized to 0-1: {value}")
    if left >= right or top >= bottom:
        raise ValueError(f"invalid bbox boundaries: {value}")
    return [round(left, 6), round(top, 6), round(right, 6), round(bottom, 6)]


def normalize_coordinate(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"coordinate must be a two-number list: {value}")
    try:
        x, y = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"coordinate values must be numbers: {value}") from exc
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return [round(x, 6), round(y, 6)]


def get_field_value(case: dict[str, Any], key: str) -> Any:
    for item in case.get("input", {}).get("fields", []):
        if item.get("key") == key:
            return item.get("value")
    return None


def extract_a22_reference_fdis(case: dict[str, Any]) -> list[int]:
    for item in case.get("reference_answer", {}).get("fields", []):
        if item.get("key") == "teeth":
            return sorted(normalize_fdi(tooth.get("fdi")) for tooth in item.get("value", []))
    return []


def extract_a23_target_fdis(case: dict[str, Any]) -> list[int]:
    for source_key in ("input", "reference_answer"):
        for item in case.get(source_key, {}).get("fields", []):
            if item.get("key") not in {"target_fdis", "target_teeth", "teeth"}:
                continue
            value = item.get("value", [])
            if not isinstance(value, list):
                continue
            fdis: list[int] = []
            for tooth in value:
                raw_fdi = tooth.get("fdi") if isinstance(tooth, dict) else tooth
                fdi = normalize_fdi(raw_fdi)
                if fdi not in fdis:
                    fdis.append(fdi)
            if fdis:
                return fdis
    if is_counterfactual_case(case):
        return generate_counterfactual_a23_fdis(case)
    return []


def extract_a24_reference_structures(case: dict[str, Any]) -> list[dict[str, Any]]:
    for item in case.get("reference_answer", {}).get("fields", []):
        if item.get("key") == "structures":
            value = item.get("value", [])
            return value if isinstance(value, list) else []
    return []


def extract_a24_cephalometric_structures(case: dict[str, Any]) -> list[str]:
    for item in case.get("input", {}).get("fields", []):
        if item.get("key") != "target_structures":
            continue
        value = item.get("value", [])
        if isinstance(value, list):
            structures = [normalize_a24_structure_code(code) for code in value]
            return list(dict.fromkeys(structures))

    structures = []
    for item in extract_a24_reference_structures(case):
        if not isinstance(item, dict):
            continue
        structure_code = normalize_a24_structure_code(item.get("structure_code"))
        if structure_code not in structures:
            structures.append(structure_code)
    return structures


def extract_a24_periapical_target_fdis(case: dict[str, Any]) -> list[int]:
    for source_key in ("input", "reference_answer"):
        for item in case.get(source_key, {}).get("fields", []):
            if item.get("key") not in {"target_fdis", "target_teeth", "teeth"}:
                continue
            value = item.get("value", [])
            if not isinstance(value, list):
                continue
            fdis: list[int] = []
            for tooth in value:
                raw_fdi = tooth.get("fdi") if isinstance(tooth, dict) else tooth
                fdi = normalize_fdi(raw_fdi)
                if fdi not in fdis:
                    fdis.append(fdi)
            if fdis:
                return sorted(fdis, key=lambda fdi: FDI_LEFT_TO_RIGHT_RANK[fdi])

    fdis = []
    for item in extract_a24_reference_structures(case):
        if not isinstance(item, dict) or "fdi" not in item:
            continue
        fdi = normalize_fdi(item["fdi"])
        if fdi not in fdis:
            fdis.append(fdi)
    return sorted(fdis, key=lambda fdi: FDI_LEFT_TO_RIGHT_RANK[fdi])


def select_a24_cephalometric_structures(case: dict[str, Any]) -> list[str]:
    """Select four landmarks deterministically, preserving the case list order."""
    structures = extract_a24_cephalometric_structures(case)
    if len(structures) <= 4:
        return structures
    rng = _case_stable_rng(case)
    indexes = sorted(rng.sample(range(len(structures)), 4))
    return [structures[index] for index in indexes]


def select_a24_periapical_target_fdis(case: dict[str, Any]) -> list[int]:
    """Select one target tooth deterministically from the case's target teeth."""
    fdis = extract_a24_periapical_target_fdis(case)
    if len(fdis) <= 1:
        return fdis
    # case237 has a fixed annotation target; do not let the seeded sampling
    # select a different tooth for this case.
    if case.get("case_code") == "case237" and 31 in fdis:
        return [31]
    rng = _case_stable_rng(case)
    return [fdis[rng.randrange(len(fdis))]]


def extract_a24_periapical_image_fdis(case: dict[str, Any]) -> list[int]:
    for item in case.get("input", {}).get("fields", []):
        if item.get("key") not in {"image_fdis", "visible_fdis", "included_fdis"}:
            continue
        value = item.get("value", [])
        if isinstance(value, list):
            fdis = {normalize_fdi(fdi) for fdi in value}
            return sorted(fdis, key=lambda fdi: FDI_LEFT_TO_RIGHT_RANK[fdi])
    return extract_a24_periapical_target_fdis(case)


def extract_a24_periapical_left_to_right_fdis(case: dict[str, Any]) -> list[int]:
    """Return image FDI values in the fixed dental arch order."""
    return extract_a24_periapical_image_fdis(case)


def is_counterfactual_case(case: dict[str, Any]) -> bool:
    for source_key in ("input", "reference_answer"):
        for item in case.get(source_key, {}).get("fields", []):
            if item.get("key") == "counterfactual" and item.get("value") is True:
                return True
    return False


def generate_counterfactual_a23_fdis(case: dict[str, Any]) -> list[int]:
    seed_source = str(case.get("id") or case.get("input", {}).get("images", [{}])[0].get("path", ""))
    rng = random.Random(seed_source)
    return rng.sample(sorted(VALID_FDI_CODES), 2)


def normalize_action_value(value: Any) -> str:
    if isinstance(value, dict):
        action_code = str(value.get("action_code", "")).strip()
        requested_view = str(value.get("requested_view", "")).strip()
    else:
        action_code = str(value).strip()
        requested_view = ""

    action_code = OLD_ACTION_CODE_TO_LABEL.get(action_code, action_code)
    if action_code == "补充缺失视角":
        return f"{action_code}:{requested_view}"
    return action_code


def format_clinical_purpose(purpose: dict[str, Any]) -> str:
    custom_goal_text = str(purpose.get("custom_goal_text", "")).strip()
    if purpose.get("goal_code") == "OTHER_REVIEW" and custom_goal_text:
        return custom_goal_text

    site = purpose.get("site", {})
    site_text = "、".join(site.get("region_codes", [])) if isinstance(site, dict) else ""
    department_code = purpose.get("department_code", "")
    goal_code = purpose.get("goal_code", "")
    parts = [
        DEPARTMENT_LABELS.get(department_code, department_code),
        GOAL_LABELS.get(goal_code, goal_code),
    ]
    if site_text:
        parts.append(f"部位：{site_text}")
    return "；".join(part for part in parts if part)


def clinical_input_fields(case: dict[str, Any]) -> dict[str, Any]:
    """Keep structured values (including numbers and objects) intact."""
    return {item["key"]: item["value"] for item in case["input"].get("fields", [])}


def clinical_diagnosis_options(case: dict[str, Any]) -> list[dict[str, Any]]:
    options = clinical_input_fields(case).get("diagnosis_options", [])
    if case["task_code"] in {"A5.1", "A6.2"}:
        return _fixed_choice_options(case)
    if case["task_code"] != "A5.2":
        return options
    # The candidate set is defined by the doctor's recorded option IDs. Roles
    # and reasons must never enter the evaluated model's prompt.
    entries = next((f["value"] for f in case["reference_answer"]["fields"] if f["key"] == "differential_diagnoses"), [])
    ids = [entry["option_id"] for entry in entries]
    available = [option["option_id"] for option in options]
    if not ids or len(set(ids)) != len(ids) or len(set(available)) != len(available) or set(ids) - set(available):
        raise ValueError("invalid A5.2 doctor-recorded candidate IDs")
    return [option for option in options if option["option_id"] in set(ids)]


def _fixed_choice_options(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Build six deterministic choices: all correct options plus distractors."""
    task_code = case["task_code"]
    options = clinical_input_fields(case).get("diagnosis_options", [])
    entries = next(
        (
            field.get("value")
            for field in case.get("reference_answer", {}).get("fields", [])
            if field.get("key") == "selected_option_ids"
        ),
        [],
    )
    correct_ids = [str(option_id).strip() for option_id in (entries or []) if str(option_id).strip()]
    if len(correct_ids) != len(set(correct_ids)):
        raise ValueError(f"duplicate {task_code} correct option IDs")
    if len(correct_ids) > 6:
        raise ValueError(f"{task_code} has more than six correct option IDs: {correct_ids}")

    by_id = {str(option.get("option_id", "")).strip(): option for option in options}
    if len(by_id) != len(options) or set(correct_ids) - set(by_id):
        raise ValueError(f"invalid {task_code} correct option IDs")
    distractors = [option for option in options if option.get("option_id") not in set(correct_ids)]
    if len(correct_ids) + len(distractors) < 6:
        raise ValueError(f"{task_code} must have at least six diagnosis options")

    rng = _case_stable_rng(case)
    rng.shuffle(distractors)
    selected = [by_id[option_id] for option_id in correct_ids] + distractors[: 6 - len(correct_ids)]
    rng.shuffle(selected)
    return selected


def clinical_media(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve M identifiers by path, in the actual image sending order."""
    evidence = clinical_input_fields(case).get("evidence_media", [])
    by_path = {item["path"]: item for item in evidence}
    images = case["input"].get("images", [])
    paths = [item["path"] for item in images]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate image paths in clinical case")
    if set(by_path) != set(paths) or len(by_path) != len(evidence):
        raise ValueError("evidence_media must match input.images one to one")
    ordered = [by_path[path] for path in paths]
    ids = [item["evidence_id"] for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence_media identifiers")
    return ordered


def build_clinical_prompt(case: dict[str, Any]) -> str:
    """Render the supplied templates from inputs only, never reference answers."""
    code = case["task_code"]
    if code not in CLINICAL_FIELD_KEYS:
        raise ValueError(f"unsupported clinical task: {code}")
    fields = clinical_input_fields(case)
    media = clinical_media(case)
    facts = "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in fields.get("clinical_facts", [])
    ) or "未额外提供"
    options = "、".join(item["diagnosis"] for item in clinical_diagnosis_options(case))
    target_entity = fields.get("target_entity", "未提供")
    entity_description = fields.get("entity_description")
    if code == "A6.1" and entity_description:
        target_entity = f"{target_entity}（{entity_description}）"
    replacements = {
        "科室中文名称": DEPARTMENT_LABELS.get(fields.get("department_code"), fields.get("department_code", "未提供")),
        "事实列表": facts,
        "事实列表；未提供时注明未额外提供": facts,
        "所选科室对应的诊断列表": options,
        "本题正确诊断与待排除诊断组成的中文候选列表": options,
        "诊断名称": fields.get("diagnosis", fields.get("target_diagnosis", "未提供")),
        "诊断目标": fields.get("diagnostic_goal", "未提供"),
        "实体名称": target_entity,
        "按发送顺序列出的影像M编号": (
            "、".join(item["evidence_id"] for item in media)
            if code == "A6.1"
            else "\n".join(item["evidence_id"] for item in media)
        ),
        "性别": fields.get("sex", "未提供"),
        "年龄": fields.get("age", "未提供"),
        "患者资料": fields.get("patient_info", "未额外提供"),
        "主诉": fields.get("chief_complaint", "未提供"),
        "病史": fields.get("history", "未提供"),
        "口腔检查结果": fields.get("clinical_exam", "未提供"),
        "影像检查描述": fields.get("imaging_exam", "未提供"),
        "患者偏好": fields.get("patient_preferences") or "未提供",
    }
    prompt = render_prompt_file(code, replacements)
    # All candidates come from input; favored/excluded labels stay private.
    if code in {"A5.1", "A5.2", "A6.2"} and not options:
        raise ValueError("diagnosis_options must not be empty")
    return prompt


# Keep the public protocol surface task-specific, like the A1-A4 tasks.  The
# shared renderer above still owns the common placeholder handling, while each
# task now has an explicit prompt builder for registration and testing.
def build_a51_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A5.1"))


def build_a52_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A5.2"))


def build_a53_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A5.3"))


def build_a61_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A6.1"))


def build_a62_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A6.2"))


def build_a63_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A6.3"))


def build_a71_prompt(case: dict[str, Any]) -> str:
    return build_clinical_prompt(_require_task(case, "A7.1"))


def _require_task(case: dict[str, Any], task_code: str) -> dict[str, Any]:
    if case.get("task_code") != task_code:
        raise ValueError(f"expected {task_code} case, got {case.get('task_code')}")
    return case


def _clinical_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _clinical_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("counts must be non-negative integers")
    return value


def parse_clinical_answer(text: str | None, task_code: str) -> dict[str, Any]:
    obj = parse_json_object(text)
    allowed = set(CLINICAL_FIELD_KEYS[task_code])
    required = allowed - ({"reason"} if task_code == "A5.3" else set())
    if set(obj) - allowed or required - set(obj):
        raise ValueError(f"invalid answer keys for {task_code}: {sorted(obj)}")
    if task_code in {"A5.1", "A5.2", "A6.2"}:
        diagnoses = obj["diagnoses"]
        if not isinstance(diagnoses, list) or not diagnoses:
            raise ValueError("diagnoses must be a non-empty array")
        obj["diagnoses"] = [_clinical_text(item, "diagnosis") for item in diagnoses]
        if len(set(obj["diagnoses"])) != len(diagnoses):
            raise ValueError("duplicate diagnoses")
        obj["reason"] = _clinical_text(obj["reason"], "reason")
        if task_code == "A5.1" and len(obj["reason"]) > 100:
            raise ValueError("A5.1 reason must be at most 100 characters")
        if task_code == "A5.2":
            obj["exclude_reason"] = _clinical_text(obj["exclude_reason"], "exclude_reason")
            if len(obj["exclude_reason"]) > 100 or len(obj["reason"]) > 100:
                raise ValueError("A5.2 reasons must be at most 100 characters each")
        if task_code == "A6.2" and len(obj["reason"]) > 100:
            raise ValueError("A6.2 reason must be at most 100 characters")
    elif task_code == "A5.3":
        if obj["sufficiency"] == "不充足":
            obj["sufficiency"] = "不足"
        if obj["sufficiency"] not in ("充足", "不足"):
            raise ValueError("sufficiency must be 充足 or 不足")
        if obj["sufficiency"] == "充足":
            if "reason" in obj:
                raise ValueError("omit reason when sufficiency is 充足")
        else:
            obj["reason"] = _clinical_text(obj.get("reason"), "reason")
            if len(obj["reason"]) > 100:
                raise ValueError("A5.3 reason must be at most 100 characters")
    elif task_code == "A6.1":
        obj["total"] = _clinical_count(obj["total"])
        if not isinstance(obj["images"], list) or not obj["images"]:
            raise ValueError("images must be a non-empty array")
        ids = []
        for item in obj["images"]:
            if not isinstance(item, dict) or set(item) != {"image", "count"}:
                raise ValueError("each image count requires image and count")
            item["image"] = _clinical_text(item["image"], "image")
            item["count"] = _clinical_count(item["count"])
            ids.append(item["image"])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate image identifiers")
        counts = [item["count"] for item in obj["images"]]
        if not max(counts) <= obj["total"] <= sum(counts):
            raise ValueError("deduplicated total must be between largest image count and sum")
    elif task_code == "A6.3":
        if set(obj) != {"category", "condition_code"}:
            raise ValueError("A6.3 requires exactly category and condition_code")
        obj["category"] = _clinical_text(obj["category"], "category")
        obj["condition_code"] = _clinical_text(obj["condition_code"], "condition_code")
        if obj["category"] not in ("患者信息", "检查状况", "补充图像"):
            raise ValueError("invalid condition category")
        if len(obj["condition_code"]) > 20:
            raise ValueError("A6.3 condition_code must be at most 20 characters")
    elif task_code == "A7.1":
        if not isinstance(obj["plans"], list) or not obj["plans"]:
            raise ValueError("plans must be a non-empty array")
        for item in obj["plans"]:
            if not isinstance(item, dict) or set(item) != {"plan"}:
                raise ValueError("invalid plans entry")
            item["plan"] = _clinical_text(item["plan"], "plan")
            if len(item["plan"]) > 100:
                raise ValueError("A7.1 plan must be at most 100 characters")
    return obj


def parse_a51_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A5.1")


def parse_a52_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A5.2")


def parse_a53_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A5.3")


def parse_a61_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A6.1")


def parse_a62_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A6.2")


def parse_a63_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A6.3")


def parse_a71_answer(text: str | None) -> dict[str, Any]:
    return parse_clinical_answer(text, "A7.1")


def validate_clinical_answer(pred: dict[str, Any], case: dict[str, Any]) -> None:
    """Check answer identifiers against this case, without consulting labels."""
    if "diagnoses" in pred:
        options = {item["diagnosis"] for item in clinical_diagnosis_options(case)}
        if set(pred["diagnoses"]) - options:
            raise ValueError("diagnoses contain names outside input options")
    if case["task_code"] == "A6.1":
        ids = [item["evidence_id"] for item in clinical_media(case)]
        if [item["image"] for item in pred["images"]] != ids:
            raise ValueError("image counts must match all input M identifiers in sending order")


def clinical_reference_answer(case: dict[str, Any]) -> dict[str, Any]:
    """Adapt native annotations to prompt output names; never used in prompts."""
    fields = {item["key"]: item["value"] for item in case["reference_answer"]["fields"]}
    code = case["task_code"]
    if code in {"A5.1", "A5.2", "A6.2"}:
        options = {item["option_id"]: item["diagnosis"] for item in clinical_input_fields(case)["diagnosis_options"]}
        if code == "A5.2":
            entries = fields["differential_diagnoses"]
            return {
                "diagnoses": [options[item["option_id"]] for item in entries if item["role"] == "favored"],
                "reason": "；".join(item["reason"] for item in entries if item["role"] == "favored"),
                "exclude_reason": "；".join(f"{options[item['option_id']]}：{item['reason']}" for item in entries if item["role"] == "excluded"),
            }
        return {"diagnoses": [options[key] for key in fields["selected_option_ids"]], "reason": fields["diagnostic_reason"]}
    if code == "A6.1":
        counts = {item["evidence_id"]: item["count"] for item in fields["image_counts"]}
        return {"total": fields["total_count"], "images": [
            {"image": item["evidence_id"], "count": counts[item["evidence_id"]]} for item in clinical_media(case)
        ]}
    if code == "A6.3":
        categories = {"patient_information": "患者信息", "examinations": "检查状况", "additional_images": "补充图像"}
        conditions = fields["required_conditions"]
        if len(conditions) != 1:
            raise ValueError("A6.3 reference must contain exactly one required condition")
        item = conditions[0]
        return {"category": categories[item["category"]], "condition_code": item["condition_code"]}
    if code == "A7.1":
        return {"plans": fields["treatment_plans"]}
    if code == "A5.3":
        mapping = {"supported": "充足", "not_supported": "不足"}
        if fields.get("diagnosis_support") not in mapping:
            raise ValueError("unknown A5.3 diagnosis_support label")
        return {**fields, "sufficiency": mapping[fields["diagnosis_support"]],
                "reason": "；".join(item["statement"] for item in fields.get("reasoning", []))}
    return fields


def build_clinical_mock_response(case: dict[str, Any]) -> str:
    """Synthetic format examples for dry-run, not clinical ground truth."""
    code = case["task_code"]
    if code in {"A5.1", "A5.2", "A6.2"}:
        response = {"diagnoses": [clinical_diagnosis_options(case)[0]["diagnosis"]], "reason": "仅用于验证输出格式。"}
        if code == "A5.2":
            response["exclude_reason"] = "仅用于验证输出格式。"
    elif code == "A5.3":
        response = {"sufficiency": "不足", "reason": "仅用于验证输出格式。"}
    elif code == "A6.1":
        response = {"total": 0, "images": [{"image": item["evidence_id"], "count": 0} for item in clinical_media(case)]}
    elif code == "A6.3":
        response = {"category": "患者信息", "condition_code": "测试信息"}
    else:
        response = {"plans": [{"plan": "仅用于验证输出格式。"}]}
    return json.dumps(response, ensure_ascii=False)

