from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.prompt_item import BenchmarkRun, PromptItem
from src.benchmark.schemas import get_schema
from src.oracles import make_oracle
from src.runners.base_runner import BaseRunner


def run_prompt(
    prompt: PromptItem,
    old_runner: BaseRunner,
    new_runner: BaseRunner,
) -> BenchmarkRun:
    """Run one PromptItem on both models, apply oracle, return BenchmarkRun."""
    oracle = make_oracle(prompt.oracle)

    if prompt.schema_name:
        # Instructor path — tests schema adherence (H2)
        schema = get_schema(prompt.schema_name)
        old_result = old_runner.run_structured(prompt.prompt_text, prompt.id, schema)
        new_result = new_runner.run_structured(prompt.prompt_text, prompt.id, schema)
    else:
        # Text path — tests format/instruction following
        old_result = old_runner.run_text(prompt.prompt_text, prompt.id)
        new_result = new_runner.run_text(prompt.prompt_text, prompt.id)

    old_passed, old_evidence = oracle.check(old_result)
    new_passed, new_evidence = oracle.check(new_result)

    return BenchmarkRun(
        prompt_id=prompt.id,
        task_family=prompt.task_family,
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
        old_passed=old_passed,
        new_passed=new_passed,
        old_evidence=old_evidence,
        new_evidence=new_evidence,
        migration_failed=(old_passed and not new_passed),
        migration_improved=(not old_passed and new_passed),
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

            # Write immediately — crash-safe
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
