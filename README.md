# DentXpertBench Evaluation

DentXpertBench-Eval is a lightweight evaluation toolkit for the DentXpertBench oral multimodal benchmark. The runner supports A1.1–A4.3 and the seven supplied A5.1–A7.1 prompt templates (A7.2 is not registered). A5.1–A5.3 support task-specific scoring and independently configured reason judging; other clinical open answers remain pending review. See [A5–A7 使用说明](docs/clinical-prompts.md).

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

模型回答和评分现在分为两个步骤。运行模型时，每个 task 会保存一个回答文件，默认命名为
`results/<task>-<model_name>.jsonl`，其中包含 `raw_response`、`thinking`、`parsed_answer`、调用参数和 token usage：

```bash
python run.py --model-config configs/model-qwen-local.json --task A1.1
python run.py --model-config configs/model-qwen-local.json --task A1.1 --case-code case35
```

之后对回答文件评分，结果保存为同名加 `-eval` 的文件：

```bash
python evaluate.py results/a1_1-Qwen-Qwen3-5-9B.jsonl
```

`--case-code` 和 `--id` 都可重复使用，分别按 `case_code` 或 `id` 只重新回答指定 case，也可以同时使用。

Dry-run every registered task:

```bash
python run.py --task all --dry-run --limit 2
```

统计 `tokenizer.json` 对所有题目文本输入的 token 数，并按 Qwen2.5-VL 常见的 28px 合并 patch 规则估算图片 token（不调用模型）：

```powershell
python count_input_tokens.py
```

结果写入 `results/input_token_counts.json` 和 `results/input_token_counts.csv`。输出同时包含文本 token、估算图片 token 和估算总输入 token。图片估算公式为每张图 `ceil(width/28) × ceil(height/28) + 2`，其中 `+2` 是 vision_start/vision_end；实际服务端会受视觉处理器的缩放策略影响。可只统计一个任务：

```powershell
python count_input_tokens.py --task A7.2
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

当前 Qwen profile 直连 `http://124.222.246.189:24758/v1`，不需要启动 SSH 隧道；默认并发为 8，生成 token 上限为 65536。若服务端的 65536 指的是上下文窗口，它仍由服务端配置控制，客户端的 `max_tokens` 只控制最大输出长度。

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

## Defaults

- A1.1 cases: `tasks/A1.1/DentXpertBench-cases-v3.2-A1.1-2026-10-10.json`
- A1.3 cases: `tasks/A1.3/DentXpertBench-cases-v3.2-A1.3-2026-10-11.json`
- Image root: `../公开数据集/公开数据集`
- A1.1 results: `results/a1_1_eval_results.jsonl`
- A1.3 results: `results/a1_3_eval_results.jsonl`

Do not store the API key itself in `configs/model.json`; keep it in the configured key file or environment variable.

