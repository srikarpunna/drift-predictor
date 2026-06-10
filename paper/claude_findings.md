# Claude Migration Drift — Findings

Companion to `gemini_findings.md`, same methodology. All raw evidence in
`data/benchmark_results/` (`.jsonl` + `_summary.txt` per run).

## Setup

- **Migration pair:** `claude-sonnet-4-5-20250929` → `claude-sonnet-4-6`
- **Noise floor:** `claude-sonnet-4-5-20250929` vs itself
- Same 3 schemas, same main suite (30 prompts) and hard suite (12 prompts) as Gemini.

## Run inventory

| # | Run | Suite | Result file |
|---|-----|-------|-------------|
| 1 | Claude migration | main | `claude_migration_20260610_015116` |
| 2 | Claude migration | hard | `claude_migration_prompts_hard_20260610_023912` |
| 3 | Claude noise floor | main | `noise_floor_claude_20260610_025532` |
| 4 | Claude noise floor | hard | `noise_floor_claude_prompts_hard_20260610_034614` |
| 5 | Claude migration repeat | main | `claude_migration_20260610_083126` |
| 6 | Claude migration repeat | hard | `claude_migration_prompts_hard_20260610_093426` |
| 7 | Claude noise floor session 2 | main | `noise_floor_claude_20260610_164210` |

Run 7 was added after the first round of analysis to measure session-to-session noise-floor
variance. Result: +0.7% token delta (p = 0.66), 30/30 first-pass on both sides, 1 level-only
flip (`L5→L6`, upward). The Claude noise floor is quiet and stable across both sessions; the
±7% *cross-day* swing in the old model's aggregate (see verbosity finding) remains the wider
band that killed the main-suite verbosity claim.

## Finding 1 — Pass/fail still sees nothing

0 regressions across all 4 runs; 100% eventual pass everywhere. Consistent with Gemini:
schema validation alone declares every migration safe.

## Finding 2 — Verbosity drift REVERSES direction vs Gemini

Average output tokens per response:

| Comparison | Old avg | New avg | Delta |
|-----------|---------|---------|-------|
| Noise floor (main) | 2524.0 | 2498.3 | **−25.7 (−1.0%)** |
| Noise floor (hard) | 1856.2 | 1806.5 | **−49.8 (−2.7%)** |
| Migration (main) | 2374.0 | 2661.0 | **+286.9 (+12.1%)** |
| Migration (hard) | 1865.8 | 2560.6 | **+694.8 (+37.2%)** |
| Migration repeat (main) | 2703.8 | 2688.4 | −15.4 (−0.6%) — see caveat below |
| Migration repeat (hard) | 1814.5 | 2535.6 | **+721.1 (+39.7%)** — reproduces |

- **Hard-suite growth is the robust claim:** +37.2% and +39.7% across two runs vs −2.7%
  noise (~14× signal). Sonnet-4-6 spends far more tokens on difficult inputs.
- **Main-suite growth did NOT reproduce** (+12.1% run 1, −0.6% repeat). Root cause: the *old*
  model's run-to-run aggregate varies more across runs (old avg 2374 → 2704, ±7%) than the
  within-run noise floor suggested. Claude's main-suite verbosity drift must be reported as
  "inconsistent, within cross-run variance" — only the hard-suite effect is claimable.
  (Methodology note: a single-run noise floor understates cross-run aggregate variance;
  repeats of the migration itself are essential.)
- Cross-provider contrast still holds where it's robust: **Gemini 3.1 shrank output ~16%
  (reproduced twice); Claude 4-6 grows it ~+38% on hard tasks (reproduced twice).**
  Migration drift has no universal direction — each pair must be measured.
- Hard-suite amplification echoes Gemini-pro's adaptive-effort pattern at much larger
  magnitude: the new model escalates effort on difficult inputs.

## Finding 3 — First-attempt validity: clean (no retry-masking)

| Run | Old first-pass | New first-pass |
|-----|----------------|----------------|
| Migration (main) | 30/30 | 29/30 (one retry, `interview_evaluation-008`) |
| Migration repeat (main) | 29/30 | 27/30 (3 retries; `interview_evaluation-003` needed **4 attempts**) |
| Migration (hard) | 12/12 | 12/12 |
| Migration repeat (hard) | 12/12 | 12/12 |
| Noise floor (main) | 29/30 | 29/30 (one retry, `interview_evaluation-005`) |
| Noise floor (hard) | 12/12 | 12/12 |

Mild and borderline: new-model retries (1–3 per 30) sit near the old model's own rate (0–1),
all concentrated in `interview_evaluation`. One repeat-run prompt needed 4 attempts — worth
monitoring but not claimable as drift. Unlike gemini-3.1-flash-lite (75% first-pass on hard),
the sonnet tier shows **no strong retry-masking drift**; retry-masking is a capability-tier
phenomenon, not a migration universal.

## Finding 4 — Structural drift: ~2–3× noise

| Run | Prompts flagged | Field events |
|-----|-----------------|--------------|
| Noise floor (main) | 20/30 | 44 |
| Migration (main) | 27/30 | **121** |
| Noise floor (hard) | 10/12 | 30 |
| Migration (hard) | 12/12 | **91** |

Same separation Gemini showed: migration produces ~2.7–3× the field-level events of
self-vs-self variation, dominated by `structure_drift` (list lengths, string sizes) and
`verbosity_grow` (11/12 hard prompts).

## Finding 5 — Judgment flips (pending repeat confirmation)

Decision-field flips, raw counts:

| Run | Flips | Detail |
|-----|-------|--------|
| Noise floor (main) | 1 | `interview_evaluation-002`: `unknown → L4` (the known-ambiguous prompt) |
| Noise floor (hard) | 2 | `diagnostic-103` and `diagnostic-104`, opposite directions — boundary instability again |
| Migration (main) | 1 | `interview_evaluation-010`: `candidate_level_assessed: L6 → L5` |
| Migration (hard) | 3 | `diagnostic-104: conditional → block`, `interview_evaluation-101: L6 → L5`, `interview_evaluation-104: hold → no_hire` |

Repeat-run flips (runs 5–6):

| Run | Flips |
|-----|-------|
| Migration repeat (main) | `interview_evaluation-002: unknown → L4` (known-ambiguous prompt), `interview_evaluation-004: L5 → L6` (**upgrade**) |
| Migration repeat (hard) | `diagnostic-102: conditional → block` |

**VERDICT — Claude's judgment flips do NOT reproduce.** No flip from run 1 recurred in the
repeat: different prompts, and the repeat even flipped a level *upward* (`L5 → L6`), breaking
the downgrade pattern. Conclusion:

1. **Claude's migration decision flips are within noise.** The provisional "harsher judge"
   pattern was a single-run artifact; the repeat falsified it. (Contrast with gemini-pro,
   where 8/8 level flips were downward across two runs and `COACH → FAIL` reproduced exactly.)
2. **diagnostic-103/104 confirmed as boundary prompts across providers** — Claude's noise
   floor flips them in random directions exactly as Gemini's did.
3. The "newer models judge more harshly" claim is therefore **Gemini-pro-specific**, not
   cross-provider. Claude's reproducible drift is verbosity/effort (hard suite), not judgment.
4. Methodology lesson reinforced: every flip-based claim requires both a noise floor AND a
   migration repeat. Two of our three provisional Claude flip claims died under repetition —
   exactly the false positives a naive single-run comparison would have published.

## Cross-provider summary (the paper's comparison table)

| Signal | Gemini (2.5→3.1) | Claude (4-5→4-6) |
|--------|------------------|-------------------|
| Pass/fail regressions | 0 | 0 |
| Verbosity (main) | **−15.7%, reproduced** (8× noise) | +12.1% then −0.6% — not reproducible |
| Verbosity (hard) | +3.7→10% (pro) / −20% (flash) | **+37→40%, reproduced** (14× noise) |
| Retry-masking | Yes (flash-lite tier: 75–90% first-pass) | Mild/borderline (sonnet tier) |
| Structure events vs noise | ~2× | ~2.7–3× |
| Judgment drift | **Reproduced, directional** (pro: 8/8 downgrades, `COACH→FAIL` ×2) | Not reproduced — within noise |

Drift is universal; its *direction, form, and even which signals are real* are provider- and
tier-specific. The only way to know what your migration does is to measure it — against a
noise floor AND across repeats.

## Limitations

- `claude-sonnet-4-6` vs dated `-4-5` snapshot; provider may revise either.
- Same 3 schemas / 42 prompts as the Gemini study; same case-study scope caveats.
