# PromptPost-Mortem: LLM Drift Predictor

Research framework measuring prompt and schema drift during LLM model version migrations.

## Setup

### Prerequisites
- Python 3.11+
- Google AI Studio API key (free tier)
- Anthropic API key

### Install

```bash
pip install -e ".[dev]"
```

> If `google-generativeai` is installed, uninstall it first — conflicts with `google-genai`:
> ```bash
> pip uninstall google-generativeai
> ```

### Configure

```bash
cp .env.example .env
# Edit .env — add GEMINI_API_KEY and ANTHROPIC_API_KEY
```

### Run smoke tests

```bash
# All integration tests (hits real APIs)
pytest tests/test_runners.py -v -s -m integration

# Rate-limited? Test one model per provider:
pytest -k "old" -m integration -v -s
```

## Model Pairs

| Provider  | Old Model                    | New Model               |
|-----------|------------------------------|-------------------------|
| Google    | gemini-2.5-flash             | gemini-3.1-flash-lite   |
| Anthropic | claude-sonnet-4-5-20250929   | claude-sonnet-4-6       |

## Project Structure

```
src/
  models.py          # RunResult, MigrationResult (shared data models)
  runners/           # GeminiRunner, ClaudeRunner (symmetric interface)
  oracles/           # Pass/fail detection (parser, unit-test, rubric)
  agents/            # Diagnostic + repair agents
  utils/             # Config, prompt generator, token counter
tests/
  fixtures/          # Cached real API responses for offline unit tests
data/
  synthetic_prompts/ # Generated benchmark prompts
  public_real_prompts/
  benchmark_results/
```
