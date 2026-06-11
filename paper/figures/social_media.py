"""Generate a LinkedIn-friendly version of the first-attempt validity chart.

Usage:  python3 paper/figures/social_media.py
Output: paper/figures/social_media.png

Same data as fig3_first_attempt, but: selected runs only (noise floor for
contrast + the migrations that moved), 4:3 aspect for mobile feeds, and
fonts sized to stay readable as a feed thumbnail.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "benchmark_results"
OUT = Path(__file__).resolve().parent


def load(name):
    return [json.loads(line) for line in open(RESULTS / name)]


def first_pass_pct(records, side):
    valid = sum(1 for r in records if r[f"{side}_first_attempt_valid"])
    return valid / len(records) * 100


RUNS = [
    ("Same model\nvs itself", "noise_floor_20260609_175927.jsonl"),
    ("Flash mig\n(easy tasks)", "gemini_migration_20260609_172936.jsonl"),
    ("Flash mig\n(hard tasks)", "gemini_migration_prompts_hard_20260609_184220.jsonl"),
    ("Pro mig", "gemini_pro_migration_20260609_232303.jsonl"),
    ("Claude mig", "claude_migration_20260610_015116.jsonl"),
]


def main():
    labels, old_vals, new_vals = [], [], []
    for label, fname in RUNS:
        recs = load(fname)
        labels.append(label)
        old_vals.append(first_pass_pct(recs, "old"))
        new_vals.append(first_pass_pct(recs, "new"))
        print(f"  {label.replace(chr(10), ' '):28s} old {old_vals[-1]:5.1f}%  new {new_vals[-1]:5.1f}%")

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.bar([i - w / 2 for i in x], old_vals, w, label="Old model", color="#607d8b")
    ax.bar([i + w / 2 for i in x], new_vals, w, label="New model", color="#ef6c00")
    for i, (o, n) in enumerate(zip(old_vals, new_vals)):
        ax.text(i - w / 2, o + 0.8, f"{o:.0f}", ha="center", fontsize=15, fontweight="bold")
        ax.text(i + w / 2, n + 0.8, f"{n:.0f}", ha="center", fontsize=15, fontweight="bold")
    ax.axhline(100, color="#9e9e9e", lw=1.2, ls="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Got the output right on the first try (%)", fontsize=14)
    ax.set_ylim(60, 106)
    ax.tick_params(axis="y", labelsize=12)
    fig.suptitle("Every migration “passed”. The retries tell another story.",
                 fontsize=19, fontweight="bold", y=0.97)
    ax.set_title(
        "Schema validation reported zero regressions in all 19 runs.\n"
        "The gap between bars is failures the retry layer silently fixed.",
        fontsize=13, color="#444444", pad=12)
    ax.annotate("1 in 4 hard tasks\nfailed first try",
                xy=(2 + w / 2, 75), xytext=(2.75, 65),
                fontsize=14, color="#bf360c", ha="left", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bf360c", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#bf360c", lw=2))
    ax.legend(loc="lower left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "social_media.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT / 'social_media.png'}")


if __name__ == "__main__":
    main()
