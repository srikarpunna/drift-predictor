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
    fig.suptitle("How much longer or shorter did answers get after the migration?",
                 fontsize=13, y=0.985)
    ax.set_title(
        "Each bar is one full benchmark run. Gray = old model compared against itself "
        "(pure randomness).\nA migration signal is real only if it escapes the gray band "
        "AND shows up again in the repeat run.",
        fontsize=9, color="#444444", pad=10)
    ax.set_xlim(min(deltas) - 8, max(deltas) + 8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=GEMINI),
        plt.Rectangle((0, 0), 1, 1, color=CLAUDE),
    ]
    ax.legend(handles, ["Noise floor (same model both sides)",
                        "Gemini migration", "Claude migration"],
              loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    ax.set_ylim(60, 108)
    fig.suptitle("How often did the model get the output right on the FIRST try?",
                 fontsize=13, y=0.985)
    ax.set_title(
        "Final pass rate is 100% in every run, so normal testing sees nothing. "
        "The gap between bars is failures\nthat the retry layer silently fixed: "
        "extra latency and cost that never show up in success metrics.",
        fontsize=9, color="#444444", pad=10)
    ax.annotate("new flash model: 1 in 4 hard\nrequests needed a retry rescue",
                xy=(3 + w / 2, 75), xytext=(4.15, 66),
                fontsize=8.5, color="#bf360c", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bf360c", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#bf360c"))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig3_first_attempt.png", dpi=200)
    plt.close(fig)


def fig2_funnel():
    survived = [
        "Gemini-pro output shrank 15.7%\n(showed up in BOTH runs, 8x bigger than noise)",
        "Claude grew 37-40% on hard tasks\n(showed up in BOTH runs, 14x bigger than noise)",
        "New flash model fails 1 in 4 hard first tries\n(old model: almost never)",
        "Gemini-pro judges candidates lower\n(8 of 8 changes were downgrades, twice)",
    ]
    killed_gate1 = "\u201cHard prompts expose decision flips\u201d\nKILLED: the old model flips these\nprompts against ITSELF too"
    killed_gate2 = [
        "\u201cClaude judges more harshly\u201d\nKILLED: ran it again,\nnone of the flips came back",
        "\u201cClaude +12% longer answers\u201d\nKILLED: ran it again,\ngot \u22120.6% instead",
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axis("off")

    y_flow = 0.80

    ax.text(0.08, y_flow, "START\n\n7 suspected\ndrift signals\nafter the first\nmigration runs",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eceff1", ec="#455a64"))

    ax.text(0.35, y_flow,
            "CHECK 1: Noise floor\n\nCompare the old model\nagainst ITSELF.\nIf the \u201cdrift\u201d shows up here,\nit is just randomness.",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#f9a825"))

    ax.text(0.62, y_flow,
            "CHECK 2: Repeat\n\nRun the whole migration\ntest a second time.\nReal drift shows up again;\nrandomness does not.",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#f9a825"))

    ax.text(0.88, 0.94, "CONFIRMED: 4 real drift signals", ha="center",
            fontsize=10.5, fontweight="bold", color=GREEN)
    for i, s in enumerate(survived):
        ax.text(0.88, 0.83 - i * 0.155, s, ha="center", va="center", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.35", fc="#e8f5e9", ec=GREEN))

    for x0, x1 in [(0.155, 0.25), (0.45, 0.52), (0.72, 0.76)]:
        ax.annotate("", xy=(x1, y_flow), xytext=(x0, y_flow),
                    arrowprops=dict(arrowstyle="->", color="#455a64", lw=2))
    ax.text(0.20, y_flow + 0.06, "7 in", ha="center", fontsize=8.5, color="#455a64")
    ax.text(0.485, y_flow + 0.06, "6 pass", ha="center", fontsize=8.5, color="#455a64")
    ax.text(0.74, y_flow + 0.06, "4 pass", ha="center", fontsize=8.5, color="#455a64")

    ax.annotate("", xy=(0.35, 0.40), xytext=(0.35, 0.60),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.355, 0.50, "1 killed", fontsize=8.5, color=RED)
    ax.text(0.35, 0.28, killed_gate1, ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=RED))

    ax.annotate("", xy=(0.62, 0.44), xytext=(0.62, 0.60),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.625, 0.52, "2 killed", fontsize=8.5, color=RED)
    for i, s in enumerate(killed_gate2):
        ax.text(0.62, 0.32 - i * 0.19, s, ha="center", va="center", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec=RED))

    ax.set_title(
        "How 7 suspected drift findings became 4: each one had to pass two checks",
        fontsize=12.5, pad=12)
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
