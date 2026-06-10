"""Decision-field flip analysis across released benchmark runs.

Reproduces the paper's judgment-drift table (Section 5.6) directly from the
released JSONL files: for every run, every prompt, compares the value of each
decision field between the old and new model's output and reports flips.

Works on all released runs, including those recorded before `decision_flips`
was written into the JSONL records, because it recomputes flips from the raw
`old_output` / `new_output` strings.

Usage:  python3 scripts/analyze_flips.py [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.drift_metrics import decision_flips  # noqa: E402

RESULTS = ROOT / "data" / "benchmark_results"

# Same run inventory as scripts/compute_stats.py.
RUNS = [
    ("Flash noise floor (main)", "noise_floor_20260609_175927.jsonl"),
    ("Flash noise floor (hard)", "noise_floor_prompts_hard_20260610_164208.jsonl"),
    ("Pro noise floor (main)", "noise_floor_pro_20260610_000737.jsonl"),
    ("Pro noise floor s2 (main)", "noise_floor_pro_20260610_164951.jsonl"),
    ("Pro noise floor (hard)", "noise_floor_pro_prompts_hard_20260610_005204.jsonl"),
    ("Claude noise floor (main)", "noise_floor_claude_20260610_025532.jsonl"),
    ("Claude noise floor s2 (main)", "noise_floor_claude_20260610_164210.jsonl"),
    ("Claude noise floor (hard)", "noise_floor_claude_prompts_hard_20260610_034614.jsonl"),
    ("Flash migration r2 (main)", "gemini_migration_20260609_172936.jsonl"),
    ("Flash migration r3 (main)", "gemini_migration_20260609_182638.jsonl"),
    ("Flash migration (hard)", "gemini_migration_prompts_hard_20260609_184220.jsonl"),
    ("Pro migration (main)", "gemini_pro_migration_20260609_232303.jsonl"),
    ("Pro migration repeat (main)", "gemini_pro_migration_20260610_010202.jsonl"),
    ("Pro migration (hard)", "gemini_pro_migration_prompts_hard_20260609_235645.jsonl"),
    ("Pro migration repeat (hard)", "gemini_pro_migration_prompts_hard_20260610_013734.jsonl"),
    ("Claude migration (main)", "claude_migration_20260610_015116.jsonl"),
    ("Claude migration repeat (main)", "claude_migration_20260610_083126.jsonl"),
    ("Claude migration (hard)", "claude_migration_prompts_hard_20260610_023912.jsonl"),
    ("Claude migration repeat (hard)", "claude_migration_prompts_hard_20260610_093426.jsonl"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", action="store_true", help="List every individual flip"
    )
    args = parser.parse_args()

    header = f"{'Run':36s} {'n':>3s} {'flips':>6s} {'final-decision':>15s}"
    print(header)
    print("-" * len(header))

    for label, fname in RUNS:
        path = RESULTS / fname
        if not path.exists():
            print(f"{label:36s}  (missing: {fname})")
            continue

        n = 0
        all_flips: list[tuple[str, dict]] = []
        for line in open(path):
            r = json.loads(line)
            n += 1
            for fl in decision_flips(r["old_output"], r["new_output"]):
                all_flips.append((r["prompt_id"], fl))

        finals = [f for _, f in all_flips if f["final_decision"]]
        print(f"{label:36s} {n:3d} {len(all_flips):6d} {len(finals):15d}")
        if args.verbose:
            for pid, fl in all_flips:
                marker = "  [FINAL]" if fl["final_decision"] else ""
                print(f"    {pid:30s} {fl['field']}: {fl['old']} -> {fl['new']}{marker}")


if __name__ == "__main__":
    main()
