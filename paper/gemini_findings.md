# Gemini Migration Drift — Findings

All raw evidence lives in `data/benchmark_results/` (one `.jsonl` per run with full outputs,
plus an auto-generated `_summary.txt`). Every number below is reproducible from those files.

## Setup

- **Framework:** prompts (`.txt`) + Pydantic schemas + `instructor` (max_retries=3) per provider.
- **Schemas (3):** `diagnostic`, `interview_evaluation`, `support_audit` — all enum-like fields
  hardened to `Literal` types, plus cross-field validators.
- **Prompt suites:** main (30 prompts, 10/schema) and hard (12 prompts, 4/schema — edge cases,
  conflicting instructions, validator traps, underspecification).
- **Model pairs:**
  - Flash migration: `gemini-2.5-flash` → `gemini-3.1-flash-lite`
  - Pro migration: `gemini-2.5-pro` → `gemini-3.1-pro-preview`
  - Noise floors: each old model vs itself (baseline for natural variation)
- **Metrics beyond pass/fail:** first-attempt validity (before instructor retry), retry count,
  output tokens, field-level drift events (list-length and string-length changes), drift flags,
  decision-field flips.

## Run inventory

| # | Run | Suite | Result file (timestamp) |
|---|-----|-------|--------------------------|
| 1 | Flash noise floor (2.5-flash vs itself) | main | `noise_floor_20260609_175927` |
| 2 | Flash migration repeat 2 | main | `gemini_migration_20260609_172936` |
| 3 | Flash migration repeat 3 | main | `gemini_migration_20260609_182638` |
| 4 | Flash migration | hard | `gemini_migration_prompts_hard_20260609_184220` |
| 5 | Pro migration | main | `gemini_pro_migration_20260609_232303` |
| 6 | Pro migration | hard | `gemini_pro_migration_prompts_hard_20260609_235645` |
| 7 | Pro noise floor (2.5-pro vs itself) | main | `noise_floor_pro_20260610_000737` |
| 8 | Pro noise floor (2.5-pro vs itself) | hard | `noise_floor_pro_prompts_hard_20260610_005204` |
| 9 | Pro migration repeat 2 | main | `gemini_pro_migration_20260610_010202` |
| 10 | Pro migration repeat 2 | hard | `gemini_pro_migration_prompts_hard_20260610_013734` |

## Finding 1 — Pass/fail evaluation sees nothing

Every run, every model, every suite: **0 regressions**. Both old and new models eventually
produce schema-valid output on 100% of prompts. A standard "does it parse / does it validate"
eval would conclude all these migrations are safe. Everything below is what that conclusion misses.

## Finding 2 — Retry-masking: the new flash model fails first attempts

First-attempt validity (output valid *before* instructor's error-feedback retry):

| Run | Old first-pass | New first-pass | Rescued by retry |
|-----|---------------|----------------|------------------|
| Flash migration r2 (main) | 29/30 (96.7%) | 25/30 (83.3%) | 5 (4× interview_evaluation, 1× support_audit) |
| Flash migration r3 (main) | 30/30 (100%) | 27/30 (90.0%) | 3 |
| Flash migration (hard) | 12/12 (100%) | **9/12 (75.0%)** | 3 |
| Flash noise floor | 30/30 (100%) | 29/30 (96.7%) | 1 |
| Pro migration (main + hard) | 100% | 100% | 0 |

- Failures concentrate in the most constraint-dense schema (`interview_evaluation`) and get
  worse on the hard suite (75% first-pass).
- Auto-retry **hides** this completely from pass/fail metrics — in production this is extra
  latency, extra cost, and a model running closer to its failure boundary.
- The pro tier does not exhibit this failure mode at all; it drifts in *judgment* instead (Finding 5).

## Finding 3 — Verbosity drift: 8× above the noise floor

Average output tokens per response (main suite, 30 prompts):

| Comparison | Old avg | New avg | Delta |
|-----------|---------|---------|-------|
| Flash noise floor (self vs self) | 1532.7 | 1554.5 | **+21.9 (+1.4%)** |
| Flash migration r2 | 1505.8 | 1328.4 | −177.3 (−11.8%) |
| Flash migration r3 | 1552.3 | 1322.8 | −229.6 (−14.8%) |
| Pro noise floor (self vs self) | 1588.9 | 1557.6 | **−31.3 (−2.0%)** |
| Pro migration | 1576.8 | 1330.0 | −246.8 (−15.7%) |
| Pro migration repeat 2 | 1585.3 | 1337.0 | −248.3 (−15.7%) — **identical to run 1** |

Self-vs-self variation is ±2%. Both migrations shrink output by 12–16% — roughly **8× the
noise floor**. The 3.1 generation is systematically terser, and this is a migration effect,
not sampling randomness. Repeats (r2 vs r3) show the magnitude is stable across runs.

Structural drift shows the same separation: pro noise floor flags 16/30 prompts (43 field
events) vs pro migration 26/30 (83 events) — ~2× prompts, ~2× events.

## Finding 4 — Adaptive effort: hard prompts reverse the verbosity pattern (pro only)

| Run | Suite | Token delta |
|-----|-------|-------------|
| Pro migration | main | −246.8 avg (−15.7%) |
| Pro migration | hard | **+36.3 avg (+3.7%)** |
| Pro migration repeat 2 | hard | **+96.5 avg (+10.1%)** — reproduces |
| Flash migration | hard | −227.5 avg (−20.5%) — shrinkage persists |

The new pro model is terser on routine tasks but spends *more* tokens on hard ones —
consistent with adaptive effort allocation. The flash-lite model shrinks indiscriminately.
Implication for practitioners: verbosity drift measured on easy prompts does not predict
behavior on hard ones; both suites are needed.

## Finding 5 — Judgment drift: same valid schema, different decisions

Decision-field flips (`recommended_action`, `overall_recommendation`,
`candidate_level_assessed`, `overall_grade`, `escalation_required`):

| Run | Suite | Flips | Final-decision flips (rec/grade/action) |
|-----|-------|-------|------------------------------------------|
| Flash noise floor | main | 1 (`L6→L5` level only) | 0 |
| Pro noise floor | main | 2 (both `unknown↔L4` on prompts 002/003) | **0** |
| Pro noise floor | **hard** | 3 | **3** (`conditional→block`, `block→conditional`, `hold→no_hire`) |
| Flash migration r3 | main | 4 | 1 (`hold → hire`!) |
| Flash migration | hard | 7 | 4 (2× `block → conditional_proceed`, `no_hire → hold`, `escalation: True → False`) |
| Pro migration | main | 4 (all candidate-level) | 0 |
| Pro migration | hard | 3 | 3 (`conditional → block`, `hold → no_hire`, `COACH → FAIL`) |

Key observations:

1. **On the main suite, models never flip their own final decisions.** Across 60 self-vs-self
   main-suite prompts, zero changes in hire/no-hire, pass/fail, proceed/block. The only
   self-flips are `candidate_level_assessed` on prompts 002/003 — inherently ambiguous;
   their migration flips should be discounted.
2. **CRITICAL CAVEAT — hard prompts are decision-unstable even within a model.** The
   hard-suite noise floor (run 8) shows 2.5-pro flips 3/12 of its *own* final decisions,
   the same rate as the migration, on overlapping prompts (diagnostic-104,
   interview_evaluation-104). The hard prompts were designed to sit on decision boundaries —
   so raw flip *counts* on the hard suite measure boundary instability, **not** migration
   drift. Naive decision-flip counting on adversarial prompts overstates drift; a
   same-suite noise floor is mandatory. (This is itself a methodology finding.)
3. **Directional consistency CONFIRMED by repeats (runs 9–10).** Noise-floor flips are
   random in direction; migration flips are not:
   - **Candidate levels (main suite):** 8 flips across 2 migration runs, **all 8 downward**
     (`L6→L5`, `L5→unknown`, `L4→unknown`); noise-floor flips went *upward* (`unknown→L4`).
     Prompts 003 and 005 reproduce their exact flips in both runs. The new pro model
     systematically assesses candidates lower.
   - **support_audit-104 (hard):** `COACH → FAIL` in *both* migration runs, never flips in
     the noise floor — the single cleanest judgment-drift datapoint in the study.
   - **interview_evaluation-104 (hard):** `hold → no_hire` in both migration runs, but the
     old model also flips this one against itself — claim only weakly.
   - **diagnostic-104 (hard):** correctly discarded as noise (flips both directions in
     noise floor; migration flip did not reproduce).
   - Direction differs by tier: new pro is *harsher* (rejects, fails, downgrades levels);
     new flash-lite is *more lenient* (un-blocks, downgrades escalations, `hold → hire`).
4. Pro candidate-level flips on main-suite prompts 004/005 (`L6 → L5`) never occur in the
   noise floor — a genuine re-calibration of seniority judgment by the new model.

## Finding 6 — Noise floor is mandatory methodology

Without runs 1 and 7, Findings 3–5 would be unfalsifiable ("maybe the model is just random").
With them: token signal is 8× noise, structure signal ~2× noise, final-decision flips are
3-vs-0. Any drift framework must ship with a self-vs-self baseline.

## Gap-filler runs — all complete (2026-06-10)

1. **Pro noise floor on hard suite (run 8): DONE.** Hard prompts ARE inherently
   decision-unstable (3 self-flips). Hard-suite flip *counts* are noise; only
   direction-consistent, reproducible flips count as drift.
2. **Pro migration repeat, main suite (run 9): DONE.** Token shrinkage identical
   (−248.3 vs −246.8). 4 candidate-level flips again, all downward; 003/005 reproduce exactly.
3. **Pro migration repeat, hard suite (run 10): DONE.** `COACH→FAIL` and `hold→no_hire`
   reproduce exactly; adaptive-effort token increase reproduces (+10.1%).

Gemini evidence collection is complete. Next: Claude pair for provider generality.

## Known limitations (state these in the paper)

- 3 schemas, 30 + 12 prompts — a case study, not a broad benchmark.
- Gemini-only so far; Claude pair (`claude_migration`) runs next to show provider generality.
- `gemini-3.1-pro-preview` is a preview model; behavior may change at GA.
- Single provider SDK path (`instructor`); retry-masking findings depend on its retry semantics.

## One-line summary

> Both models pass 100% of schema validation across every run; meanwhile the migration
> silently shrinks output 16% (8× the noise floor), fails 25% of hard first attempts
> (flash), and re-calibrates seniority judgments the old model never wavers on — while
> our hard-suite noise floor shows that naive decision-flip counting on adversarial
> prompts overstates drift, making same-model baselines a mandatory part of any
> migration eval.
