# Paper Brief — for the writing agent

This document is the complete instruction set for drafting the paper. The quantitative
evidence lives in `gemini_findings.md` and `claude_findings.md` (every number there was
extracted from the raw `.jsonl` files in `data/benchmark_results/`). This file adds the
narrative, structure, claims discipline, and methodology details.

## Working titles (pick or improve)

- "Your Migration Passed Every Test: Detecting Silent Behavioral Drift in LLM Upgrades"
- "Drift or Dice? Separating Real Behavioral Change from Sampling Noise in LLM Migrations"
- "Schema-Valid but Different: A Noise-Floored Framework for LLM Migration Testing"

## The one-paragraph pitch

Teams are forced to migrate LLMs constantly (deprecations, pricing, new releases). The
standard safety check — run the eval suite, confirm outputs still validate — passed 100%
of the time across 16 benchmark runs of two real migrations (Gemini 2.5→3.1, Claude
sonnet-4-5→4-6). Underneath those green checkmarks: one migration silently shrank output
16%, the other grew it 40% on hard tasks, and one systematically downgraded human
candidates in interview evaluations. Just as important: roughly half of the drift we
initially "detected" turned out to be sampling noise — caught only because the framework
ships with same-model noise floors and migration repeats. We release the framework so any
team can run prompts + schemas through it and get decision-grade drift reports.

## Narrative arc (use this structure inside Results/Discussion)

1. **The blindness:** 16 runs, 2 providers, 0 regressions. Pass/fail sees nothing. Ever.
2. **The finer lens:** first-attempt validity, retries, token deltas, field-level structure
   changes, decision-field flips — drift appears everywhere.
3. **The doubt:** is it drift or dice? Introduce the two controls: (a) noise floor = old
   model vs itself; (b) repeat every migration run.
4. **The executions (be explicit, this is the paper's credibility):**
   - "Hard prompts expose decision flips" → DIED. 2.5-pro flips 3/12 of its own decisions
     on the hard suite; the prompts are boundary-unstable by design.
   - "Claude judges candidates more harshly" → DIED. Zero flips reproduced; repeat even
     flipped a level upward.
   - "Claude +12% verbosity on main suite" → DIED. Run 2 showed −0.6%; the old model's own
     cross-run aggregate varies ±7%.
5. **The survivors (each reproduced ≥2×, each ≫ noise):**
   - Gemini −15.7% output tokens, twice (−246.8 / −248.3 avg tok); noise ±2%. ~8× signal.
   - Claude +37.2% → +39.7% output tokens on hard suite; noise −2.7%. ~14× signal.
   - Gemini-pro judgment recalibration: 8/8 candidate-level flips downward across 2 runs
     (L6→L5, L5→unknown, L4→unknown); noise-floor flips go upward. `COACH→FAIL` support
     grade reproduced exactly twice; never flips in noise floor.
   - Retry-masking on the cheap tier: gemini-3.1-flash-lite first-pass 75–90% (down from
     ~100%), all rescued silently by instructor retries; sonnet tier clean. Tier effect.
6. **The lesson:** drift is universal but its direction/form is provider- and tier-specific
   (Gemini shrank; Claude grew; pro got harsher; lite got more lenient on single runs).
   There is no rule of thumb. Measure your own pair, against a noise floor, with repeats.

## Claims discipline (hard rules for the writer)

- NEVER present a killed finding as real; present them as the false positives the method caught.
- Every surviving claim must cite: effect size, noise-floor size, and reproduction count.
- Hard-suite decision-flip *counts* must never be cited as drift (boundary instability).
- "Newer models judge more harshly" is Gemini-pro-specific. Do NOT generalize cross-provider.
- The Claude main-suite verbosity result is "inconsistent across runs" — report it as a
  finding about cross-run aggregate variance, not as drift.
- Keep the limitations section verbatim-honest: 3 schemas, 30+12 prompts, 2 providers,
  preview-model caveat (gemini-3.1-pro-preview), single SDK path (instructor).

## Methodology details (for the Methods section)

- **Framework:** Python; prompts as `.txt` files; output contracts as Pydantic schemas
  (auto-discovered by filename); `instructor` library enforces schema with max_retries=3
  and error-feedback; providers: Gemini (google-generativeai), Anthropic.
- **Schemas (3):** `diagnostic` (QA migration report; `recommended_action` Literal),
  `interview_evaluation` (panel scoring; cross-field validators; `overall_recommendation`,
  `candidate_level_assessed` Literals), `support_audit` (call audit; `overall_grade`,
  `escalation_required`). Enum-like fields hardened to `typing.Literal` (previously
  decorative descriptions).
- **Suites:** main = 30 prompts (10/schema, realistic); hard = 12 prompts (4/schema:
  edge cases, conflicting instructions, validator traps, underspecification).
- **Metrics per run:** eventual pass/fail; first-attempt validity (via instructor
  PARSE_ERROR hooks, before retry); attempt counts; output tokens; field-level drift events
  (list-length `__len` and string-length `__chars` deltas, threshold 40%); drift flags
  (verbosity ±25%, structure, content shrink, first-attempt fail); decision-field flips.
- **Controls:** noise floor = identical model both sides, same suite; repeats = full
  re-run of each migration configuration.
- **Pairs:** gemini-2.5-flash→gemini-3.1-flash-lite; gemini-2.5-pro→gemini-3.1-pro-preview;
  claude-sonnet-4-5-20250929→claude-sonnet-4-6. Run dates: 2026-06-09/10.
- **Total runs:** 16 (3 flash migration repeats, flash hard, flash noise floor, 2× pro
  migration main, 2× pro migration hard, pro noise floor main+hard, 2× claude migration
  main, 2× claude migration hard, claude noise floor main+hard).

## Paper structure

1. Abstract — the one-paragraph pitch above
2. Introduction — forced migrations; the green-checkmark fallacy; contributions list
3. Related work — eval frameworks, regression testing for LLMs, behavioral drift studies
   (writer: do a literature pass; we have not done one)
4. Framework — architecture, metrics, the two controls
5. Experimental setup — methodology details above
6. Results — narrative arc steps 1–2, 5 (survivors with tables from findings docs)
7. The false positives — narrative arc steps 3–4 (own section; this is the differentiator)
8. Discussion — practitioner guidance: what to run before any migration; cost/latency
   implications; judgment drift in human-impacting pipelines
9. Limitations — list above
10. Conclusion

## Target venue & tone

- arXiv preprint first; then an LLM-evaluation / deployment workshop.
- Tone: practitioner-honest, evidence-first, no hype. The paper's authority comes from
  the false positives it admits, not from inflated claims.
- Companion artifact: the GitHub repo (cleaned, README quickstart). The paper should
  reference prompts/schemas/results as reproducible artifacts.

## Strongest single sentences (use them)

- "Across 16 benchmark runs of two production migrations, schema-level validation flagged
  zero regressions; meanwhile output volume shifted up to 40% and one migration
  systematically downgraded its assessments of human candidates."
- "Half of the drift we initially detected was sampling noise — a same-model noise floor
  and one repeat run were sufficient to falsify it."
- "Drift is universal; its direction is not."
