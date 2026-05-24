"""
Run the 30-prompt pilot benchmark on both Gemini models.

Usage:
    python3.11 scripts/run_pilot.py

Output:
    data/benchmark_results/pilot_gemini.jsonl  — one BenchmarkRun per line
    data/benchmark_results/pilot_summary.txt   — human-readable summary
"""

import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_prompts.pilot.prompts import PILOT_PROMPTS
from src.benchmark.benchmark_runner import load_results, run_benchmark
from src.runners.gemini_runner import GeminiRunner
from src.utils.config import GeminiModels, settings

OUTPUT_JSONL = Path("data/benchmark_results/pilot_gemini.jsonl")
OUTPUT_SUMMARY = Path("data/benchmark_results/pilot_summary.txt")


def print_summary(runs) -> str:
    lines = []
    total = len(runs)
    regressions = [r for r in runs if r.migration_failed]
    improvements = [r for r in runs if r.migration_improved]
    old_pass = sum(1 for r in runs if r.old_passed)
    new_pass = sum(1 for r in runs if r.new_passed)

    lines.append("=" * 60)
    lines.append("PILOT BENCHMARK SUMMARY")
    lines.append(f"Models: {runs[0].old_model_id}  →  {runs[0].new_model_id}")
    lines.append("=" * 60)
    lines.append(f"Total prompts:      {total}")
    lines.append(f"Old model pass:     {old_pass}/{total}  ({old_pass/total*100:.1f}%)")
    lines.append(f"New model pass:     {new_pass}/{total}  ({new_pass/total*100:.1f}%)")
    lines.append(f"Regressions:        {len(regressions)}  (old✓ new✗)")
    lines.append(f"Improvements:       {len(improvements)}  (old✗ new✓)")
    lines.append("")

    # By task family
    lines.append("BY TASK FAMILY")
    lines.append("-" * 40)
    by_family = defaultdict(list)
    for r in runs:
        by_family[r.task_family].append(r)

    for family, family_runs in sorted(by_family.items()):
        n = len(family_runs)
        fails = sum(1 for r in family_runs if r.migration_failed)
        old_p = sum(1 for r in family_runs if r.old_passed)
        new_p = sum(1 for r in family_runs if r.new_passed)
        lines.append(f"{family:20s}  n={n}  old={old_p}/{n}  new={new_p}/{n}  regressions={fails}")

    lines.append("")

    # Token analysis
    lines.append("TOKEN ANALYSIS (output tokens)")
    lines.append("-" * 40)
    old_toks = [r.old_tokens_out for r in runs]
    new_toks = [r.new_tokens_out for r in runs]
    lines.append(f"Old model avg:  {sum(old_toks)/len(old_toks):.1f} tokens/response")
    lines.append(f"New model avg:  {sum(new_toks)/len(new_toks):.1f} tokens/response")
    lines.append(f"Delta:          {(sum(new_toks)-sum(old_toks))/len(runs):+.1f} avg tokens/response")
    lines.append("")

    # Regressions detail
    if regressions:
        lines.append("REGRESSIONS (old✓ → new✗)")
        lines.append("-" * 40)
        for r in regressions:
            lines.append(f"  {r.prompt_id:15s}  [{r.task_family}]")
            lines.append(f"    evidence: {r.new_evidence}")

    summary = "\n".join(lines)
    return summary


def main():
    print(f"Loading {len(PILOT_PROMPTS)} pilot prompts...")

    old_runner = GeminiRunner(model_id=GeminiModels.OLD, settings=settings)
    new_runner = GeminiRunner(model_id=GeminiModels.NEW, settings=settings)

    print(f"Running: {GeminiModels.OLD}  vs  {GeminiModels.NEW}")
    print(f"Output:  {OUTPUT_JSONL}")
    print()

    runs = run_benchmark(
        prompts=PILOT_PROMPTS,
        old_runner=old_runner,
        new_runner=new_runner,
        output_path=OUTPUT_JSONL,
        verbose=True,
    )

    summary = print_summary(runs)
    print()
    print(summary)

    OUTPUT_SUMMARY.write_text(summary)
    print(f"\nSummary saved → {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
