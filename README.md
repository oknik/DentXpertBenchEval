# DentXpertBench Evaluation

DentXpertBench-Eval is a lightweight evaluation toolkit for the DentXpertBench oral multimodal benchmark. The runner supports A1.1–A4.3 and the seven supplied A5.1–A7.1 prompt templates (A7.2 is not registered). A5.1–A5.3, A6.1, A6.2, and A7.1 use approved task-specific scoring; A6.3 remains pending review. See [A5–A7 使用说明](docs/clinical-prompts.md).

## Project Layout

```text
DentXpertBench-Eval/
├── README.md
├── pyproject.toml
├── prepare.py
├── run.py
├── evaluate.py
├── bench/
│   ├── convert.py
│   ├── media.py
│   ├── protocol.py
│   ├── models.py
│   └── metrics/
├── tasks/                      # task specs and task case JSON files
├── configs/
├── tests/
├── data/                       # evaluation run data
└── results/
```

The previous single-task evaluation quickstart is kept in `docs-eval-legacy.md`.

## Quick Start

Dry-run the first two A1.1 cases without calling a model:

```bash
python run.py --task A1.1 --dry-run --limit 2
```

Dry-run A1.3:

```bash
python run.py --task A1.3 --dry-run --limit 2
```

Dry-run every registered task:

```bash
python run.py --task all --dry-run --limit 2
```

在 Windows 下也可以直接双击项目根目录的 `Run-All-A1-A4.cmd`，一次按顺序运行全部 A1-A4 任务。正式运行前可先在 PowerShell 中执行：

```powershell
.\Run-All-A1-A4.ps1 -DryRun -Limit 1
```

如果要使用 `configs` 目录中的三个 API profile 依次跑完全部 14 个任务，可执行：

```powershell
.\Run-All-Three-APIs.ps1
```

结果会直接写入现有的 `results/` 目录，并沿用现有命名风格，同时在时间戳后加入模型名，例如 `a1_1_eval_results_时间戳_Qwen-Qwen3-5-9B.jsonl`，避免三个模型的结果互相覆盖。正式运行前建议先执行 `.\Run-All-Three-APIs.ps1 -DryRun -Limit 1`；如需失败后继续，可加 `-ContinueOnError`。

脚本默认使用 `configs/model-gpt-6-luna.json`；如需切换模型，可传入 `-ModelConfig configs/model-chatanywhere.json`。

Run an OpenAI-compatible multimodal model:

```bash
python run.py
```

The model, endpoint, and generation settings are configured in a selected JSON profile. The default profile is `configs/model.json`, which uses Sublyx; the API key is read from `D:\Desktop\sublyx.txt` or `SUBLYX_API_KEY`.

For OpenAI-compatible Qwen servers, use the included local profile:

```powershell
python run.py --model-config configs/model-qwen-local.json --task A1.1 --limit 1
```

Each model result records the provider-reported `token_usage` (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `visible_completion_tokens`, and `total_tokens`) in the JSONL file. The console also prints usage per case. The `.summary.json` file contains aggregate usage including average, maximum, and P95 total tokens, plus reasoning-token totals. If the server does not return an OpenAI-compatible `usage` object, the field is left empty and the console reports that usage is unavailable.

To tune `max_tokens`, remember that Qwen reasoning tokens are usually counted inside `completion_tokens`. If the server reports `reasoning_tokens`, use `p95_completion_tokens` plus headroom; that budget covers both thinking and visible answer tokens. `max_tokens` limits generated output tokens; it does not limit prompt tokens.

To switch manually to the additional `gpt-6-luna` profile:

```bash
python run.py --model-config configs/model-gpt-6-luna.json
```

The Luna profile uses the Sublyx OpenAI-compatible endpoint and key by default. If Luna is provided by another API service, change `base_url`, `api_key_file`, and/or `api_key_env` only in `configs/model-gpt-6-luna.json`; the original profile remains unchanged. Command-line options such as `--temperature` and `--max-tokens` still override profile defaults.

Recompute metrics from saved model answers:

```bash
python evaluate.py results/a1_1_eval_results.jsonl
python evaluate.py results/a1_3_eval_results.jsonl
```

## A6.1 / A6.2 / A7.1 使用与计分（中文速览）

题库规模：A6.1 30 题 70 张图，A6.2 34 题 80 张图，A7.1 60 题纯文字病例。

```powershell
# 试运行：只构造提示词、读取影像并校验格式，不调用任何 API，不写结果文件
python run.py --task A6.1 --dry-run --limit 1 --image-root 'D:\你的数据集根目录'
python run.py --task A6.2 --dry-run --limit 1 --image-root 'D:\你的数据集根目录'
python run.py --task A7.1 --dry-run --limit 1

# 正式运行：被测模型配置沿用 --model-config；独立评分模型默认 configs/model-gpt-6-luna.json
python run.py --task A6.1 --image-root 'D:\你的数据集根目录' --model-config configs/model-deepseek-flash.json
python run.py --task A6.2 --image-root 'D:\你的数据集根目录' --model-config configs/model-deepseek-flash.json
python run.py --task A7.1 --model-config configs/model-deepseek-flash.json
```

| 任务 | 计分（每题 100 分） | 评分方式 |
| --- | --- | --- |
| A6.1 | 去重总数完全正确 50 分；逐图计数每张完全正确分摊 50 分（匹配张数 / 题面影像数） | 纯客观，不调用评分模型 |
| A6.2 | 诊断集合 F1 × 50 分；诊断理由独立评分 0/1/2 → 0/25/50 分 | 一次独立评分请求，传入医生/模型诊断及理由 |
| A7.1 | 独立评分 0/1/2 → 0/50/100 分 | 一次独立评分请求，整体比较模型计划数组、医生参考计划数组和患者资料 |

- A6.1 的 `total` 与逐图 `count` 必须是非负严格整数（布尔值、浮点、负数均无效）；M 编号必须与题面影像一一对应且顺序一致，重复、缺失、多余或乱序均判为无效；去重总数须在最大单图数与逐图之和之间。无效答案 `case_score` 为 `null`，不按 0 分计入。
- A6.2 的诊断必须来自题面选项且不重复，理由不超过 100 字；评分模型同时看到医生/模型诊断和理由，不自行补充评分标准；核心事实或诊断角色颠倒时理由为 0。
- A7.1 只接受纯文字病例；`plans` 为非空 `[{"plan": "..."}]`。评分按整体含义而非数组项数机械匹配，允许备选方案、同义表述及步骤拆分/合并/顺序等价；参考计划可能不完整，评分不得虚构参考要求或声称经过专家验证。
- 评分模型调用或解析失败时，已算出的客观分保留，理由/计划分及该题总分保持 `null`，状态为待审核；不会默认为 0 分。A6.1 永不调用评分模型。
- `--dry-run` 不调用被测模型或评分模型，也不写结果文件。`results/`、`preview/` 中的运行产物不应提交。
- 重新汇总已有结果：`python evaluate.py results/实际文件.jsonl --task A6.1`。`evaluate.py` 对 A6.1/A6.2/A7.1 从已保存的 `model_answer` 重算客观分，并复用已保存的 `reason_judgment`，不调用任何 API；已完成的人工/评分结果不会静默丢失。
- A6.3 本次未改动，仍保持待审核状态。

## Defaults

- A1.1 cases: `tasks/A1.1/DentXpertBench-cases-v3.2-A1.1-2026-10-10.json`
- A1.3 cases: `tasks/A1.3/DentXpertBench-cases-v3.2-A1.3-2026-10-11.json`
- Image root: `../公开数据集/公开数据集`
- A1.1 results: `results/a1_1_eval_results.jsonl`
- A1.3 results: `results/a1_3_eval_results.jsonl`

Do not store the API key itself in `configs/model.json`; keep it in the configured key file or environment variable.

