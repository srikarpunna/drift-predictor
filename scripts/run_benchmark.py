"""
Run the benchmark for a given model pair.

Usage:
    python scripts/run_benchmark.py --pair gemini_migration
    python scripts/run_benchmark.py --pair claude_migration

Reads config.json from project root.
Loads all prompt sets defined in config.
Saves results to data/benchmark_results/<pair>_<timestamp>.jsonl
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.benchmark_runner import load_results, run_benchmark
from src.benchmark.loader import load_all_prompts, load_config
from src.runners.claude_runner import ClaudeRunner
from src.runners.gemini_runner import GeminiRunner
from src.utils.config import settings


def build_runners(pair, provider_settings):
    if pair.provider == "gemini":
        old = GeminiRunner(model_id=pair.old_model, settings=provider_settings)
        new = GeminiRunner(model_id=pair.new_model, settings=provider_settings)
    elif pair.provider == "claude":
        old = ClaudeRunner(model_id=pair.old_model, settings=provider_settings)
        new = ClaudeRunner(model_id=pair.new_model, settings=provider_settings)
    else:
        raise ValueError(f"Unknown provider: {pair.provider}")
    return old, new


def print_summary(runs, pair) -> str:
    lines = []
    total = len(runs)
    regressions = [r for r in runs if r.migration_failed]
    improvements = [r for r in runs if r.migration_improved]
    old_pass = sum(1 for r in runs if r.old_passed)
    new_pass = sum(1 for r in runs if r.new_passed)

    lines.append("=" * 60)
    lines.append("BENCHMARK SUMMARY")
    lines.append(f"Pair:    {pair.name}")
    lines.append(f"Old:     {pair.old_model}")
    lines.append(f"New:     {pair.new_model}")
    lines.append("=" * 60)
    lines.append(f"Total prompts:   {total}")
    lines.append(f"Old model pass:  {old_pass}/{total}  ({old_pass/total*100:.1f}%)")
    lines.append(f"New model pass:  {new_pass}/{total}  ({new_pass/total*100:.1f}%)")
    lines.append(f"Regressions:     {len(regressions)}  (old✓ → new✗)")
    lines.append(f"Improvements:    {len(improvements)}  (old✗ → new✓)")
    lines.append("")

    lines.append("BY SCHEMA")
    lines.append("-" * 40)
    by_family = defaultdict(list)
    for r in runs:
        by_family[r.schema_name].append(r)
    for family, fr in sorted(by_family.items()):
        n = len(fr)
        fails = sum(1 for r in fr if r.migration_failed)
        old_p = sum(1 for r in fr if r.old_passed)
        new_p = sum(1 for r in fr if r.new_passed)
        lines.append(f"  {family:25s}  n={n}  old={old_p}/{n}  new={new_p}/{n}  reg={fails}")

    lines.append("")
    lines.append("TOKEN ANALYSIS (output tokens)")
    lines.append("-" * 40)
    old_toks = [r.old_tokens_out for r in runs]
    new_toks = [r.new_tokens_out for r in runs]
    lines.append(f"  Old avg:  {sum(old_toks)/len(old_toks):.1f} tok/response")
    lines.append(f"  New avg:  {sum(new_toks)/len(new_toks):.1f} tok/response")
    lines.append(f"  Delta:    {(sum(new_toks)-sum(old_toks))/len(runs):+.1f} avg tok/response")
    lines.append("")

    if regressions:
        lines.append("REGRESSIONS  (old✓ → new✗)")
        lines.append("-" * 40)
        for r in regressions:
            lines.append(f"  {r.prompt_id:20s}  [{r.schema_name}]")
            lines.append(f"    evidence: {r.new_evidence}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="Model pair name from config.json")
    args = parser.parse_args()

    cfg = load_config()
    pair = cfg.get_pair(args.pair)
    prompts = load_all_prompts(cfg)

    print(f"Loaded {len(prompts)} prompts from {len(cfg.prompt_sets)} set(s)")
    print(f"Running: {pair.old_model}  →  {pair.new_model}")
    print()

    old_runner, new_runner = build_runners(pair, settings)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl = cfg.output_dir / f"{pair.name}_{timestamp}.jsonl"
    output_summary = cfg.output_dir / f"{pair.name}_{timestamp}_summary.txt"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    runs = run_benchmark(
        prompts=prompts,
        old_runner=old_runner,
        new_runner=new_runner,
        output_path=output_jsonl,
        verbose=cfg.verbose,
    )

    summary = print_summary(runs, pair)
    print()
    print(summary)
    output_summary.write_text(summary)
    print(f"\nResults → {output_jsonl}")
    print(f"Summary → {output_summary}")


if __name__ == "__main__":
    main()
