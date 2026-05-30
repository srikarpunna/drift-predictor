from __future__ import annotations

from pathlib import Path

from src.artifacts.schemas import get_schema
from src.benchmark.prompt_item import BenchmarkRun, PromptItem
from src.runners.base_runner import BaseRunner


def run_prompt(
    prompt: PromptItem,
    old_runner: BaseRunner,
    new_runner: BaseRunner,
) -> BenchmarkRun:
    """Run one PromptItem on both models, return BenchmarkRun.

    Validation is fully delegated to Pydantic — if Instructor parses the
    response into the schema successfully, all fields (including nested ones)
    already passed their constraints. No separate assertion layer needed.
    """
    if prompt.output_schema:
        schema = get_schema(prompt.output_schema)
        old_result = old_runner.run_structured(prompt.prompt_text, prompt.id, schema)
        new_result = new_runner.run_structured(prompt.prompt_text, prompt.id, schema)
    else:
        old_result = old_runner.run_text(prompt.prompt_text, prompt.id)
        new_result = new_runner.run_text(prompt.prompt_text, prompt.id)

    return BenchmarkRun(
        prompt_id=prompt.id,
        schema_name=prompt.output_schema,
        old_model_id=old_runner.model_id,
        new_model_id=new_runner.model_id,
        old_output=old_result.output_text,
        new_output=new_result.output_text,
        old_tokens_in=old_result.tokens_in,
        old_tokens_out=old_result.tokens_out,
        new_tokens_in=new_result.tokens_in,
        new_tokens_out=new_result.tokens_out,
        old_latency_ms=old_result.latency_ms,
        new_latency_ms=new_result.latency_ms,
        old_passed=old_result.succeeded,
        new_passed=new_result.succeeded,
        old_evidence=old_result.error,
        new_evidence=new_result.error,
        migration_failed=(old_result.succeeded and not new_result.succeeded),
        migration_improved=(not old_result.succeeded and new_result.succeeded),
    )


def run_benchmark(
    prompts: list[PromptItem],
    old_runner: BaseRunner,
    new_runner: BaseRunner,
    output_path: Path,
    verbose: bool = True,
) -> list[BenchmarkRun]:
    """
    Run full benchmark. Appends each result to output_path as JSONL immediately
    (crash-safe — partial results preserved if run fails midway).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkRun] = []

    with open(output_path, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            if verbose:
                print(f"[{i:3}/{len(prompts)}] {prompt.id} ... ", end="", flush=True)

            run = run_prompt(prompt, old_runner, new_runner)
            results.append(run)

            f.write(run.model_dump_json() + "\n")
            f.flush()

            if verbose:
                status = "FAIL" if run.migration_failed else ("IMPR" if run.migration_improved else "OK")
                print(
                    f"{status}  "
                    f"old={'✓' if run.old_passed else '✗'}  "
                    f"new={'✓' if run.new_passed else '✗'}  "
                    f"Δtok={run.token_delta:+d}"
                )

    return results


def load_results(path: Path) -> list[BenchmarkRun]:
    """Load JSONL results file."""
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(BenchmarkRun.model_validate_json(line))
    return runs
