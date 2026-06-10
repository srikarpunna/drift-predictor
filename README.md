# drift-predictor

A framework for detecting behavioral drift when you migrate from one LLM to another.

Pass/fail schema validation always returns 100% — this framework measures what's actually changing underneath: output volume, first-attempt failure rates, field-level structural changes, and whether high-stakes decisions (hire/no-hire, pass/fail, proceed/block) are flipping.

Includes the prompts, schemas, and raw results from a 19-run study across Gemini 2.5→3.1 and Claude sonnet-4-5→4-6. See [`paper/draft.md`](paper/draft.md) for the full write-up (PDF: [`paper/draft.pdf`](paper/draft.pdf)).

---

## Quickstart: run your own migration pair

### 1. Install

```bash
git clone https://github.com/srikarpunna/drift-predictor.git
cd drift-predictor
pip install -e ".[dev]"
```

> If you previously installed `google-generativeai`, remove it first — it conflicts with `google-genai`:
> ```bash
> pip uninstall google-generativeai
> ```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add:
#   GEMINI_API_KEY=...
#   ANTHROPIC_API_KEY=...
```

### 3. Define your model pair

Edit `config.json` and add an entry under `model_pairs`:

```json
{
  "name": "my_migration",
  "provider": "gemini",
  "old_model": "gemini-2.5-flash",
  "new_model": "gemini-3.1-flash-lite"
}
```

Supported providers: `"gemini"` and `"claude"`.

### 4. Add your prompts and schemas

**Prompts** go in `src/artifacts/prompts/` — one `.txt` file per task:

```
src/artifacts/prompts/my_task-001.txt
src/artifacts/prompts/my_task-002.txt
```

Each file is just a plain text prompt. The filename prefix (before the first `-`) is matched to a schema.

**Schemas** go in `src/artifacts/schemas/` — one `.py` file per task type, defining a Pydantic model (a Python class that specifies exactly what fields and types the model's JSON output must have):

```python
# src/artifacts/schemas/my_task.py
from pydantic import BaseModel
from typing import Literal

class MyTask(BaseModel):
    result: Literal["pass", "fail"]
    summary: str
    confidence: float
```

The schema filename (`my_task.py`) must match the prompt filename prefix (`my_task-001.txt`).

### 5. Run

```bash
python scripts/run_benchmark.py --pair my_migration
```

Results are saved to `data/benchmark_results/` as a `.jsonl` file (one record per prompt) and a `_summary.txt`.

To run against a separate folder of harder/edge-case prompts:

```bash
python scripts/run_benchmark.py --pair my_migration --prompts src/artifacts/prompts_hard
```

---

## What the output tells you

The summary report covers four things pass/fail won't show you:

**First-attempt validity** — did the model produce valid output on the first try, or did it need the retry mechanism to fix it? A model that needs retries on 25% of requests looks identical to a perfect model in your success metrics, but is silently costing you latency and API spend.

**Token delta** — how much did average output volume change? If the new model generates 15% fewer tokens on every response, your users are getting shorter answers. If it generates 40% more on complex tasks, your costs went up.

**Field-level drift** — beyond overall volume, did specific fields change? A model might keep numeric fields identical while cutting summary fields in half, or vice versa. The framework flags any field where the length changed by more than 40%.

**Decision-field flips** — for fields that carry a final decision (like `"hire"` vs `"no_hire"`, or `"proceed_migration"` vs `"block_migration"`), did the new model give a different answer on the same input? If a flip is reproducible and directional (e.g., always downgrading, never upgrading), that is not noise — it is the model recalibrating its thresholds.

---

## Noise floor: the most important run you'll skip

A single migration run cannot tell you if drift is real or just sampling randomness. Run a **noise floor** to find out: set both `old_model` and `new_model` to the same model in `config.json`, then run the benchmark.

```json
{
  "name": "noise_floor",
  "provider": "gemini",
  "old_model": "gemini-2.5-flash",
  "new_model": "gemini-2.5-flash"
}
```

```bash
python scripts/run_benchmark.py --pair noise_floor
```

Any signal the noise floor produces is natural model variation. A migration signal is only real if it exceeds this baseline. Without the noise floor, every non-deterministic model looks like it drifted.

Then **repeat the migration run** at least once. A signal that appears in run 1 but not run 2 is a false positive. In our 19-run study, this falsified three of our initial "findings" before we published anything.

---

## Analyzing results

Two analysis scripts work directly on the `.jsonl` result files:

```bash
python scripts/compute_stats.py    # paired bootstrap CIs + permutation p-values for token/latency deltas
python scripts/analyze_flips.py    # decision-field flips per run (--verbose lists each flip)
```

Both recompute everything from the raw outputs, so they work on any run, including ones recorded before a given metric existed.

---

## Included prompts and schemas

Three schemas, two prompt suites:

| Schema | Task | Prompts (main) | Prompts (hard) |
|--------|------|----------------|----------------|
| `diagnostic` | QA migration diagnostic report | 10 | 4 |
| `interview_evaluation` | Technical interview panel evaluation | 10 | 4 |
| `support_audit` | Customer support call audit | 10 | 4 |

**Main suite** (`src/artifacts/prompts/`): realistic, well-specified tasks. Good for measuring typical production behavior.

**Hard suite** (`src/artifacts/prompts_hard/`): adversarial prompts designed to sit near decision boundaries — edge cases, conflicting instructions, cross-field consistency traps. Reveals capability limits invisible in the main suite.

---

## Model pairs used in the paper

| Name in config | Old model | New model |
|----------------|-----------|-----------|
| `gemini_migration` | `gemini-2.5-flash` | `gemini-3.1-flash-lite` |
| `gemini_pro_migration` | `gemini-2.5-pro` | `gemini-3.1-pro-preview` |
| `claude_migration` | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-6` |
| `noise_floor` | `gemini-2.5-flash` | `gemini-2.5-flash` |
| `noise_floor_pro` | `gemini-2.5-pro` | `gemini-2.5-pro` |
| `noise_floor_claude` | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-5-20250929` |

Raw results for all runs are in `data/benchmark_results/`.

---

## Project structure

```
config.json                   # model pairs, prompt sets, output dir
scripts/
  run_benchmark.py            # main entry point
  compute_stats.py            # statistical analysis of token/latency deltas
  analyze_flips.py            # decision-field flip analysis
src/
  artifacts/
    prompts/                  # main suite (.txt files)
    prompts_hard/             # hard suite (.txt files)
    schemas/                  # Pydantic output schemas (.py files)
  benchmark/
    benchmark_runner.py       # runs both models, compares outputs
    drift_metrics.py          # token deltas, field-level events, flip detection
    loader.py                 # loads config, prompts, schemas
    prompt_item.py            # data model for a single prompt+schema pair
  runners/
    gemini_runner.py          # Gemini runner (uses google-genai + instructor)
    claude_runner.py          # Claude runner (uses anthropic + instructor)
    base_runner.py            # shared interface
  models.py                   # RunResult, MigrationResult
  utils/
    config.py                 # settings loader (.env → API keys)
data/
  benchmark_results/          # .jsonl + _summary.txt for each run
paper/
  draft.md                    # paper write-up
  gemini_findings.md          # Gemini evidence
  claude_findings.md          # Claude evidence
```

---

## Requirements

Python 3.11+. All dependencies in `pyproject.toml`:

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # includes pytest, ruff, mypy
```

Key dependencies: `google-genai`, `anthropic`, `instructor`, `pydantic>=2.7`.

---

## License

MIT. See `pyproject.toml`.
