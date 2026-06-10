"""Paired statistics for per-prompt output-token deltas.

For each run, treats each prompt as a paired observation (old tokens, new tokens) and reports:
  - mean per-prompt delta in tokens and as % of the old mean
  - 95% CI on the mean delta via paired bootstrap (resample prompts with replacement)
  - p-value via a sign-flip permutation test (null: no systematic direction;
    randomly flip the sign of each prompt's delta and compare |mean|)

Usage:  python3 scripts/compute_stats.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "benchmark_results"

N_BOOT = 10_000
N_PERM = 10_000
SEED = 42

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


def paired_deltas(path, new_key, old_key):
    deltas, old_vals = [], []
    for line in open(path):
        r = json.loads(line)
        deltas.append(r[new_key] - r[old_key])
        old_vals.append(r[old_key])
    return deltas, old_vals


def mean(xs):
    return sum(xs) / len(xs)


def bootstrap_ci(deltas, rng, n_boot=N_BOOT, alpha=0.05):
    n = len(deltas)
    means = sorted(
        mean([deltas[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def sign_flip_pvalue(deltas, rng, n_perm=N_PERM):
    observed = abs(mean(deltas))
    hits = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() < 0.5 else -d for d in deltas]
        if abs(mean(flipped)) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def report(metric_name, new_key, old_key, unit):
    print(f"\n=== {metric_name} ===")
    header = (f"{'Run':36s} {'n':>3s} {'mean Δ':>10s} {'Δ%':>7s} "
              f"{'95% CI (' + unit + ')':>24s} {'p (sign-flip)':>13s}")
    print(header)
    print("-" * len(header))
    for label, fname in RUNS:
        path = RESULTS / fname
        if not path.exists():
            print(f"{label:36s}  (missing: {fname})")
            continue
        # Per-run RNG: each run's CI/p-value is reproducible in isolation,
        # independent of which other runs are analyzed or in what order.
        rng = random.Random(f"{SEED}:{new_key}:{fname}")
        deltas, old_vals = paired_deltas(path, new_key, old_key)
        m = mean(deltas)
        pct = m / mean(old_vals) * 100
        lo, hi = bootstrap_ci(deltas, rng)
        p = sign_flip_pvalue(deltas, rng)
        print(f"{label:36s} {len(deltas):3d} {m:+10.1f} {pct:+6.1f}% "
              f"[{lo:+9.1f}, {hi:+9.1f}] {p:13.4f}")


def main():
    report("Output tokens per response", "new_tokens_out", "old_tokens_out", "tok")
    report("Latency per request (includes retry round-trips)",
           "new_latency_ms", "old_latency_ms", "ms")


if __name__ == "__main__":
    main()
