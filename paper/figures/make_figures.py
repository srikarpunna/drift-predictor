"""Generate the paper's figures directly from raw benchmark results.

Usage:  python3 paper/figures/make_figures.py
Outputs: paper/figures/fig1_verbosity.png, fig2_funnel.png, fig3_first_attempt.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "benchmark_results"
OUT = Path(__file__).resolve().parent

GRAY = "#9e9e9e"
GEMINI = "#4285F4"
CLAUDE = "#D97757"
GREEN = "#2e7d32"
RED = "#c62828"


def load(name):
    return [json.loads(line) for line in open(RESULTS / name)]


def token_delta_pct(records):
    old = sum(r["old_tokens_out"] for r in records) / len(records)
    new = sum(r["new_tokens_out"] for r in records) / len(records)
    return (new - old) / old * 100


def first_pass_pct(records, side):
    valid = sum(1 for r in records if r[f"{side}_first_attempt_valid"])
    return valid / len(records) * 100


# (label, file, kind) — kind drives the bar color.
RUNS = [
    ("Flash noise floor (main)", "noise_floor_20260609_175927.jsonl", "noise"),
    ("Pro noise floor (main)", "noise_floor_pro_20260610_000737.jsonl", "noise"),
    ("Pro noise floor (hard)", "noise_floor_pro_prompts_hard_20260610_005204.jsonl", "noise"),
    ("Claude noise floor (main)", "noise_floor_claude_20260610_025532.jsonl", "noise"),
    ("Claude noise floor (hard)", "noise_floor_claude_prompts_hard_20260610_034614.jsonl", "noise"),
    ("Flash migration r2 (main)", "gemini_migration_20260609_172936.jsonl", "gemini"),
    ("Flash migration r3 (main)", "gemini_migration_20260609_182638.jsonl", "gemini"),
    ("Flash migration (hard)", "gemini_migration_prompts_hard_20260609_184220.jsonl", "gemini"),
    ("Pro migration (main)", "gemini_pro_migration_20260609_232303.jsonl", "gemini"),
    ("Pro migration repeat (main)", "gemini_pro_migration_20260610_010202.jsonl", "gemini"),
    ("Pro migration (hard)", "gemini_pro_migration_prompts_hard_20260609_235645.jsonl", "gemini"),
    ("Pro migration repeat (hard)", "gemini_pro_migration_prompts_hard_20260610_013734.jsonl", "gemini"),
    ("Claude migration (main)", "claude_migration_20260610_015116.jsonl", "claude"),
    ("Claude migration repeat (main)", "claude_migration_20260610_083126.jsonl", "claude"),
    ("Claude migration (hard)", "claude_migration_prompts_hard_20260610_023912.jsonl", "claude"),
    ("Claude migration repeat (hard)", "claude_migration_prompts_hard_20260610_093426.jsonl", "claude"),
]

COLORS = {"noise": GRAY, "gemini": GEMINI, "claude": CLAUDE}


def fig1_verbosity():
    labels, deltas, colors = [], [], []
    noise_deltas = []
    for label, fname, kind in RUNS:
        d = token_delta_pct(load(fname))
        labels.append(label)
        deltas.append(d)
        colors.append(COLORS[kind])
        if kind == "noise":
            noise_deltas.append(d)
        print(f"  {label:34s} {d:+6.1f}%")

    noise_band = max(abs(d) for d in noise_deltas)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = range(len(labels))[::-1]
    ax.axvspan(-noise_band, noise_band, color=GRAY, alpha=0.18,
               label=f"Noise-floor band (±{noise_band:.1f}%)")
    ax.barh(y, deltas, color=colors, height=0.65)
    for yi, d in zip(y, deltas):
        ax.text(d + (0.8 if d >= 0 else -0.8), yi, f"{d:+.1f}%",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Average output-token change, new vs old model (%)")
    ax.set_title("Verbosity drift per run vs the same-model noise floor")
    ax.set_xlim(min(deltas) - 8, max(deltas) + 8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=GEMINI),
        plt.Rectangle((0, 0), 1, 1, color=CLAUDE),
    ]
    ax.legend(handles, ["Noise floor (same model both sides)",
                        "Gemini migration", "Claude migration"],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_verbosity.png", dpi=200)
    plt.close(fig)


def fig3_first_attempt():
    runs = [
        ("Flash noise floor\n(main)", "noise_floor_20260609_175927.jsonl"),
        ("Flash migration r2\n(main)", "gemini_migration_20260609_172936.jsonl"),
        ("Flash migration r3\n(main)", "gemini_migration_20260609_182638.jsonl"),
        ("Flash migration\n(hard)", "gemini_migration_prompts_hard_20260609_184220.jsonl"),
        ("Pro migration\n(main)", "gemini_pro_migration_20260609_232303.jsonl"),
        ("Claude noise floor\n(main)", "noise_floor_claude_20260610_025532.jsonl"),
        ("Claude migration\n(main)", "claude_migration_20260610_015116.jsonl"),
        ("Claude migration rpt\n(main)", "claude_migration_20260610_083126.jsonl"),
    ]
    old_vals, new_vals, labels = [], [], []
    for label, fname in runs:
        recs = load(fname)
        labels.append(label)
        old_vals.append(first_pass_pct(recs, "old"))
        new_vals.append(first_pass_pct(recs, "new"))
        print(f"  {label.replace(chr(10), ' '):34s} old {old_vals[-1]:5.1f}%  new {new_vals[-1]:5.1f}%")

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar([i - w / 2 for i in x], old_vals, w, label="Old model", color="#607d8b")
    ax.bar([i + w / 2 for i in x], new_vals, w, label="New model", color="#ef6c00")
    for i, (o, n) in enumerate(zip(old_vals, new_vals)):
        ax.text(i - w / 2, o + 0.6, f"{o:.0f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, n + 0.6, f"{n:.0f}", ha="center", fontsize=8.5)
    ax.axhline(100, color=GRAY, lw=0.8, ls="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("First-attempt schema validity (%)")
    ax.set_ylim(60, 106)
    ax.set_title("Retry-masking: first-attempt validity before instructor retries\n(eventual pass rate is 100% in every run)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_first_attempt.png", dpi=200)
    plt.close(fig)


def fig2_funnel():
    survived = [
        "Gemini-pro output shrink −15.7%\n(reproduced, ~8× noise)",
        "Claude hard-suite growth +37→40%\n(reproduced, ~14× noise)",
        "Flash retry-masking: 75–83% first-pass\n(noise floor: 97–100%)",
        "Gemini-pro judgment downgrades\n(8/8 downward, COACH→FAIL ×2)",
    ]
    killed = [
        "\u201cHard prompts expose decision flips\u201d\n(noise floor flips them too)",
        "\u201cClaude judges more harshly\u201d\n(0 flips recurred in repeat)",
        "\u201cClaude +12% main-suite verbosity\u201d\n(repeat: −0.6%, sign flipped)",
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axis("off")

    ax.text(0.07, 0.93, "7 provisional findings\n(after first migration runs)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eceff1", ec="#455a64"))

    ax.text(0.42, 0.93, "Control 1\nSame-model noise floor", ha="center", va="center",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#f9a825"))
    ax.text(0.42, 0.72, "Control 2\nMigration repeat run", ha="center", va="center",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#f9a825"))
    ax.annotate("", xy=(0.30, 0.93), xytext=(0.17, 0.93),
                arrowprops=dict(arrowstyle="->", color="#455a64"))
    ax.annotate("", xy=(0.42, 0.78), xytext=(0.42, 0.87),
                arrowprops=dict(arrowstyle="->", color="#455a64"))

    ax.text(0.78, 0.97, "SURVIVED (4)", ha="center", fontsize=11,
            fontweight="bold", color=GREEN)
    for i, s in enumerate(survived):
        ax.text(0.78, 0.87 - i * 0.135, s, ha="center", va="center", fontsize=8.8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#e8f5e9", ec=GREEN))

    ax.text(0.42, 0.40, "KILLED (3)", ha="center", fontsize=11,
            fontweight="bold", color=RED)
    for i, s in enumerate(killed):
        ax.text(0.42, 0.30 - i * 0.115, s, ha="center", va="center", fontsize=8.8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec=RED))

    ax.annotate("", xy=(0.62, 0.80), xytext=(0.52, 0.74),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.annotate("", xy=(0.42, 0.44), xytext=(0.42, 0.65),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

    ax.set_title("Roughly half of the initially detected drift was sampling noise",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_funnel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print("Fig 1 — verbosity deltas:")
    fig1_verbosity()
    print("Fig 3 — first-attempt validity:")
    fig3_first_attempt()
    fig2_funnel()
    print(f"Figures written to {OUT}")
