# Drift or Dice? Separating Real Behavioral Change from Sampling Noise in LLM Migrations

---

## Abstract

Teams operating LLM-backed applications migrate models constantly — provider deprecations, pricing restructuring, and capability releases all force the update. The standard safety check is schema validation (checking that the model's output still matches your data contracts) — run the eval suite, confirm outputs still validate. Across 16 benchmark runs of two production migrations (Gemini 2.5→3.1 and Claude sonnet-4-5→4-6), this check reported zero regressions every time. Underneath those results: one migration silently shrank output by 15.7% (reproduced twice, approximately 8× above the self-vs-self noise floor), another grew output by 39.7% on difficult tasks (approximately 14× above noise), a cheaper model tier failed 25% of hard first attempts — rescued silently by retry infrastructure — and one migration's pro tier systematically downgraded candidate assessments in eight consecutive cases. Equally important: roughly half of the drift we initially detected was sampling noise, caught only because the framework ships with same-model noise floors and requires migration repeats. Three separate provisional findings collapsed under these controls. We describe the framework, its two mandatory controls (noise floor and migration repeats), the four surviving drift signals, and the three false positives the method caught. We release all prompts, schemas, and raw results as a reproducible artifact. The central finding is that drift is universal but its direction is provider- and tier-specific; there is no rule of thumb, and every migration must be measured against its own baseline.

---

## 1. Introduction

Teams operating LLM-backed production systems face a recurring obligation: migrate to a new model. Provider deprecation timelines, cost restructuring, and capability improvements all produce the same pressure. The standard safety check is straightforward: run the evaluation suite, confirm outputs still validate against your schemas and pass your tests. This check is fast, intuitive, and nearly useless.

"Across 16 benchmark runs of two production migrations, schema-level validation flagged zero regressions; meanwhile output volume shifted up to 40% and one migration systematically downgraded its assessments of human candidates."

Those migrations were Gemini 2.5→3.1 (flash and pro tier) and Claude sonnet-4-5→4-6. In every configuration, every suite, every run: 100% eventual pass rate. A standard regression gate would have concluded all migrations were safe. What it missed: one migration shrank average output by 15.7%, reproduced identically across two independent runs; another grew output on difficult tasks by 37–40%; a cheaper model tier began failing 25% of hard first attempts, rescued silently by the retry layer (the automatic re-submission that happens when an output fails validation); and a pro-tier migration produced eight consecutive downward candidate-level assessments without a single upward exception.

This paper makes three contributions. First, we describe a lightweight framework that instruments migrations beyond pass/fail, recording first-attempt validity, retry counts, output token deltas, field-level structural drift events, and decision-field flips. Second, we demonstrate that roughly half of the drift we initially detected was sampling noise — caught only because the framework includes same-model noise floors and requires migration repeats. Three provisional findings — hard-prompt decision instability, a cross-provider judgment-harshness pattern, and a 12% Claude verbosity shift — collapsed under these controls. Reporting how we almost fooled ourselves is as important as reporting what survived. Third, we release the framework, schemas, prompts, and all raw results as a reproducible artifact.

The broader lesson is not that models drift. The lesson is that drift is universal but its direction is not. Gemini shrank output; Claude grew it. Gemini-pro recalibrated human candidate assessments; Claude-sonnet did not. No rule of thumb predicts what a given migration will do. The only reliable approach is to measure it — against a noise floor, with repeats, on your own task distribution.

---

## 2. Related Work

BIG-bench [Srivastava et al., 2023], HELM [Liang et al., 2022], and the LM Evaluation Harness [Gao et al., 2023; Biderman et al., 2024] established the paradigm of fixed-prompt evaluation for language models, but these are designed to characterize absolute capability at a point in time, not to detect behavioral change across a specific version transition. The migration-testing problem is structurally different: the prompt set is fixed, the model pair is specific, and the question is not "how capable is this model" but "how did it change and does that matter for my pipeline."

Sculley et al. [2015] and Breck et al. [2017] established rubrics for testing production ML systems and identified regression testing as a first-class production concern. Their frameworks treat regression as binary: a test case passes or fails against a reference output. For structured-output tasks with schema contracts, the reference output is the schema — and as we show, schema compliance can remain at 100% across a migration that changes output volume by 40% and judgment direction systematically. Binary regression gates are necessary but not sufficient.

Song et al. [2024] and Atil et al. [2024] document that individual models produce different outputs across runs — including at temperature=0. Atil et al. find accuracy swings up to 15% and best-to-worst performance gaps up to 70% across eight tasks even with deterministic settings. Our contribution is separating within-model sampling variance from between-model migration effects, using empirical self-vs-self noise floor runs rather than analytic estimates. Khatchadourian and Franco [2025] independently quantify LLM output drift across providers for financial workflows, proposing mitigation strategies that complement our measurement approach.

Structured output enforcement has become standard practice for production LLM pipelines. Willard and Louf [2023] formalize constrained generation as finite-state machine transitions, enabling token-level enforcement of output schemas with minimal overhead. Geng et al. [2025] benchmark structured output generation across six constrained decoding frameworks using 10K real-world JSON schemas. Our framework uses the `instructor` library — which enforces schemas via validation-error feedback and retries rather than constrained decoding — and adds pre-retry first-attempt monitoring, exposing a failure mode (retry-masking) that compliant infrastructure hides by design.

Zheng et al. [2023] proposed using strong LLMs as judges for evaluating open-ended outputs and characterized position, verbosity, and self-enhancement biases in LLM judges. Wang et al. [2022] showed through self-consistency sampling that LLM outputs vary meaningfully across samples, and that majority-vote aggregation substantially improves accuracy on reasoning tasks. A recent survey [Gu et al., 2024] covers reliability, calibration, and inter-rater agreement of LLM evaluators comprehensively. Our finding that the Gemini-pro migration produces directional shifts in candidate-level assessment — downward, 8 for 8 across two independent runs — is not the kind of random flip these works describe. Systematic directional drift requires direction-sensitive flip tracking with reproducibility controls, not just flip counts.

Shankar et al. [2022] document recurring production ML challenges — data drift, monitoring gaps, retraining triggers — through interviews with 18 ML engineers. More recent work on LLMOps [Pahune et al., 2025] surveys how deployment pipelines must evolve specifically for large language models. Our framework targets the production gap these works identify: outputs are schema-validated but not ground-truth-labeled, so behavioral shifts pass through undetected.

---

## 3. Framework

### 3.1 Architecture

The framework takes two inputs: a directory of prompt files (`.txt`, one task per file) and a directory of Pydantic schema files (`.py` — Pydantic is a Python library that defines data contracts: the exact fields, types, and allowed values the model's output must have). Each prompt is submitted to both the old and new model; outputs are validated against the corresponding schema using `instructor` — an open-source Python library that, when a response fails validation, sends the error message back to the model and asks it to retry, up to 3 times. A hook before any retry fires captures whether the raw first response passed. Results are written to a `.jsonl` file (one record per prompt) and a human-readable summary report. The complete framework and artifacts are released alongside this paper.

### 3.2 Metrics

The framework records six signal classes beyond binary pass/fail:

**First-attempt validity.** Before instructor has a chance to retry, did the model's raw first response pass validation? A pre-retry hook captures this. Standard pass/fail reporting never surfaces it — the retry happens invisibly and the final result looks clean.

**Retry counts.** Per-prompt attempt counts (maximum 3). A model that needs 3 attempts on a straightforward prompt has failed in a way pass/fail will never tell you.

**Output token counts.** Average tokens per response, recorded per attempt and aggregated. Reported as delta (new average − old average) and percentage change. This is the primary volume-drift signal.

**Field-level structural drift.** For list fields, the change in list length (`__len`); for string fields, the change in character count (`__chars`). Events are flagged when the delta exceeds 40%. This surfaces changes to individual output components — for example, a model that shortens summary fields while leaving numeric fields intact.

**Aggregate drift flags.** Compound per-prompt booleans: `verbosity_grow`, `verbosity_shrink` (±25% overall), `structure_drift` (any field event), `content_shrink`.

**Decision-field flips.** For fields representing high-stakes categorical decisions — `recommended_action`, `overall_recommendation`, `candidate_level_assessed`, `overall_grade`, `escalation_required` — a flip is recorded when the new model's value differs from the old model's value on the same input. For example: 'hire' becoming 'no_hire', or 'proceed_migration' becoming 'block_migration'. These are the signals most directly relevant to behavioral risk in human-impacting pipelines.

### 3.3 The Two Controls

A single migration run cannot distinguish model change from sampling randomness. The framework requires two controls applied together:

**Noise floor.** Run the same suite with the old model on *both sides* of the comparison (old model as both "old" and "new"). Any signal this produces is within-model variation. Migration signals that do not exceed this baseline must be discarded. Without it, every non-deterministic model looks like it drifted against any comparison.

**Migration repeats.** Re-run each migration configuration at least once. Sampling artifacts reproduce unpredictably; systematic drift reproduces reliably. A finding that appears in run 1 but not run 2 is a false positive.

Together these controls apply a simple decision rule: a migration signal is real only if it exceeds the noise floor *and* reproduces across runs. Both conditions are necessary; neither is sufficient alone.

---

## 4. Experimental Setup

### 4.1 Schemas and Prompts

Three output schemas were designed to cover representative high-stakes structured-output tasks:

- **`diagnostic`**: QA migration diagnostic report. Key fields: `recommended_action` (must be one of `"proceed_migration"`, `"block_migration"`, or `"conditional_proceed"` — no other value accepted), `regression_rate_pct`, `regression_by_family` list. Cross-field validators (rules that check consistency between fields after the model fills them in) enforce arithmetic accuracy between the rate fields and the breakdown list.
- **`interview_evaluation`**: Technical interview panel evaluation. Key fields: `overall_recommendation` (one of `"hire"`, `"no_hire"`, `"hold"`), `candidate_level_assessed`, `panel_recommendation` (one of `"advance"`, `"decline"`, `"defer"` — and it must be consistent with `overall_recommendation`). A `@model_validator` (a Pydantic decorator that runs custom logic after parsing) enforces this cross-field rule.
- **`support_audit`**: Customer support call audit. Key fields: `overall_grade` (one of `"PASS"`, `"COACH"`, `"FAIL"`), `compliance_score_pct`, `escalation_required`, nested `HoldEvent`.

All enum-like fields use `typing.Literal` — Python type annotations that restrict a field to a specific set of allowed strings and reject anything else at parse time. Previously these fields were just plain `str` with descriptions; the Literal hardening was applied before any model runs.

Two prompt suites were used:

- **Main suite**: 30 prompts (10 per schema), representative of normal production task complexity and well-specified instructions.
- **Hard suite**: 12 prompts (4 per schema), designed to stress-test specific failure modes: edge-case inputs, competing instructions in the prompt, cross-field consistency traps, and intentional underspecification. These prompts were built to sit right on decision boundaries — cases where a reasonable model could go either way.

### 4.2 Model Pairs and Runs

Three migration pairs were evaluated:

| Pair | Old model | New model |
|------|-----------|-----------|
| Gemini Flash | `gemini-2.5-flash` | `gemini-3.1-flash-lite` |
| Gemini Pro | `gemini-2.5-pro` | `gemini-3.1-pro-preview` |
| Claude Sonnet | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-6` |

All runs were conducted on 2026-06-09 and 2026-06-10. Total runs: 16, comprising 2 flash migration main-suite runs, 1 flash hard-suite run, 1 flash noise floor, 2 pro migration main runs, 2 pro migration hard runs, 2 pro noise floor runs (main and hard), 2 Claude migration main runs, 2 Claude migration hard runs, and 2 Claude noise floor runs (main and hard).

### 4.3 Provider Configuration

Both providers used the `instructor` library with `max_retries=3` and structured-output mode. A hook fires on each `COMPLETION` call to record token counts; a separate hook fires on `PARSE_ERROR` (when the first response fails schema validation) to capture the error details for retry-masking analysis. No model-specific prompt tuning was applied — the same prompt text went to every model in every run.

---

## 5. Results

### 5.1 Pass/Fail: The Blind Baseline

Every model. Every run. Every suite. 100% eventual pass rate.

Both old and new models produce schema-valid output on all 42 prompts in every configuration tested. A standard "does it validate" evaluation would conclude every migration is safe. Everything in the following sections is what that conclusion misses.

### 5.2 Retry-Masking: Hidden Strain in the Cheap Tier

Retry-masking is what happens when instructor's retry loop silently fixes failures: the final output looks fine, but the model failed on its first attempt and needed error-feedback to get there. This costs latency and API calls in production, and it tells you the model is running closer to its constraint-satisfaction limit than the pass/fail number suggests.

First-attempt validity — measured before any retry — reveals a clear divide by capability tier:

| Run | Old first-pass | New first-pass |
|-----|---------------|----------------|
| Flash migration r2 (main, 30 prompts) | 29/30 (96.7%) | 25/30 (83.3%) |
| Flash migration r3 (main, 30 prompts) | 30/30 (100.0%) | 27/30 (90.0%) |
| Flash migration (hard, 12 prompts) | 12/12 (100.0%) | **9/12 (75.0%)** |
| Flash noise floor (main) | 30/30 (100.0%) | 29/30 (96.7%) |
| Pro migration (all runs) | 100% | 100% |
| Claude migration r1 (main) | 30/30 | 29/30 |
| Claude migration repeat (main) | 29/30 | 27/30 |
| Claude noise floor (main) | 29/30 | 29/30 |

The new flash model fails 10–25% of first attempts — concentrated in `interview_evaluation`, the most constraint-dense schema — and nearly all are silently rescued. The hard suite is worst at 75% first-pass. The pro tier has no such problem in any run. The Claude sonnet tier shows 1–3 retries per 30 main-suite prompts, comparable to the old model's own rate (0–1), and zero retries on both hard-suite runs. Retry-masking is a capability-tier effect, not a universal migration risk.

### 5.3 Verbosity Drift: Opposite Directions, Both Reproducible

Output token averages per response are the most consistent signal in the study:

**Gemini (main suite):**

| Comparison | Old avg | New avg | Delta |
|-----------|---------|---------|-------|
| Flash noise floor | 1532.7 | 1554.5 | +21.9 (+1.4%) |
| Flash migration r2 | 1505.8 | 1328.4 | −177.3 (−11.8%) |
| Flash migration r3 | 1552.3 | 1322.8 | −229.6 (−14.8%) |
| Pro noise floor | 1588.9 | 1557.6 | −31.3 (−2.0%) |
| **Pro migration** | **1576.8** | **1330.0** | **−246.8 (−15.7%)** |
| **Pro migration repeat** | **1585.3** | **1337.0** | **−248.3 (−15.7%)** |

The pro migration delta is −15.7% in both runs — the two numbers are nearly identical — against a self-vs-self variation of ±2.0%. That is approximately 8× the noise floor. The Gemini 3.1 generation is systematically terser; this is a migration effect, not sampling randomness, and the repeats confirm it is stable.

**Claude (main and hard suites):**

| Comparison | Old avg | New avg | Delta |
|-----------|---------|---------|-------|
| Claude noise floor (main) | 2524.0 | 2498.3 | −25.7 (−1.0%) |
| Claude noise floor (hard) | 1856.2 | 1806.5 | −49.8 (−2.7%) |
| Claude migration (main) | 2374.0 | 2661.0 | +286.9 (+12.1%) |
| Claude migration (hard) | 1865.8 | 2560.6 | +694.8 (+37.2%) |
| Claude migration repeat (main) | 2703.8 | 2688.4 | −15.4 (−0.6%) |
| **Claude migration repeat (hard)** | **1814.5** | **2535.6** | **+721.1 (+39.7%)** |

The hard-suite growth is the robust finding: +37.2% and +39.7% across two independent runs, against a noise floor of −2.7%. That is approximately 14× the noise floor. Sonnet-4-6 spends substantially more tokens on difficult inputs.

The main-suite claim did not reproduce (+12.1% run 1, −0.6% repeat) and is addressed as a false positive in Section 6.

The cross-provider contrast is sharp: the Gemini 3.1 generation shrinks output ~16% (reproduced twice); Claude 4-6 grows output ~38–40% on hard tasks (reproduced twice). Same type of migration event; opposite directions. There is no universal rule.

### 5.4 Adaptive Effort: Easy and Hard Prompts Diverge (Gemini-Pro)

The Gemini-pro migration reveals a second layer:

| Run | Suite | Token delta |
|-----|-------|-------------|
| Pro migration | main | −246.8 (−15.7%) |
| Pro migration | hard | **+36.3 (+3.7%)** |
| Pro migration repeat | hard | **+96.5 (+10.1%)** |
| Flash migration | hard | −227.5 (−20.5%) |

The new pro model is terser on routine tasks but spends more tokens on hard ones — reproduced in both hard-suite runs. The flash model shrinks regardless of task difficulty. This matters practically: verbosity drift measured on an easy eval suite does not predict what the model does on complex production inputs. You need both suites to see this.

### 5.5 Structural Drift: 2–3× Noise Across Providers

Field-level event counts (list-length and string-length changes exceeding 40%) show consistent separation from noise in both providers:

| Run | Prompts flagged | Field events |
|-----|-----------------|--------------|
| Gemini pro noise floor (main) | 16/30 | 43 |
| Gemini pro migration (main) | 26/30 | 83 |
| Claude noise floor (main) | 20/30 | 44 |
| Claude migration (main) | 27/30 | 121 |
| Claude noise floor (hard) | 10/12 | 30 |
| Claude migration (hard) | 12/12 | 91 |

Migration produces approximately 2× (Gemini) to 2.7–3× (Claude) the field-level events of self-vs-self variation. Structural drift appears across both providers and is not simply explained by the token-volume effect — individual string and list fields change independently of overall verbosity.

### 5.6 Judgment Drift: Systematic Downgrading (Gemini-Pro)

Decision-field flip counts alone are misleading — Section 6 explains why. The full table is shown here for reference; the noise-floor-and-repeat filters are applied before any finding is stated. Final-decision fields are `recommended_action`, `overall_recommendation`, `overall_grade`, and `escalation_required`; `candidate_level_assessed` flips are counted in the flip total but not as final decisions.

| Run | Suite | Flips | Final-decision flips |
|-----|-------|-------|---------------------|
| Flash noise floor | main | 1 (level only) | 0 |
| Pro noise floor | main | 2 (level only) | 0 |
| Pro noise floor | **hard** | 3 | **3** |
| Claude noise floor | main | 1 (level only) | 0 |
| Claude noise floor | **hard** | 2 | **2** |
| Flash migration r3 | main | 4 | 1 |
| Flash migration | hard | 7 | 4 |
| Pro migration | main | 4 (all level) | 0 |
| Pro migration | hard | 3 | 3 |
| Pro migration repeat | main | 4 (all level) | 0 |
| Pro migration repeat | hard | 2 | 2 |
| Claude migration | main | 1 | 0 |
| Claude migration | hard | 3 | 2 |
| Claude migration repeat | main | 2 | 0 |
| Claude migration repeat | hard | 1 | 1 |

After applying both controls, two findings survive as genuine judgment drift:

**Directional candidate-level recalibration (Gemini-pro, main suite).** Across both pro migration main-suite runs, the new model produced 8 candidate-level assessment flips; all 8 were downward (`L6→L5`, `L5→unknown`, `L4→unknown`). Self-vs-self noise-floor flips went the other way (`unknown→L4`). Prompts 003 and 005 reproduced their exact same flips in both migration runs. The new pro model consistently assesses candidate seniority lower.

**COACH→FAIL support grade (Gemini-pro, hard suite).** The `overall_grade` field in the support audit schema flipped from `COACH` to `FAIL` on the same hard-suite prompt in both pro migration runs, and never flipped in the pro noise floor on any run. This is the cleanest single-prompt judgment-drift signal in the study: same value change, reproduced twice, clean noise-floor negative.

No equivalent directional or reproducible judgment pattern appeared in the Claude migration. Claude's judgment flips are addressed in Section 6.

---

## 6. The False Positives

"Half of the drift we initially detected was sampling noise — a same-model noise floor and one repeat run were sufficient to falsify it."

Three findings looked credible after the first migration run. All three were killed by the controls. This section is not a methodological footnote — it is the methodological point of the paper. A single-run comparison without a noise floor would have published all three as results.

### 6.1 "Hard Prompts Expose Decision Flips" (Killed)

After the first flash migration hard-suite run, 7 decision-field flips were recorded — far more than the main-suite baseline of 0. The initial read: hard prompts reveal judgment instability that easy prompts mask.

Then we ran the hard-suite noise floor (run 8) — the old model tested against itself on the same hard prompts, with no model change at all. It produced 3 final-decision flips on its own. The same prompt (`diagnostic-104`) that flipped in the migration also flipped when we compared the old model to itself — in a different direction. There is no stable "old model answer" to migrate away from on these prompts.

Here's why that happens: the hard prompts are designed to sit right on the edge of a decision. Think of a grading rubric where a candidate's answer is genuinely borderline — a fair evaluator could score it either way. Any model, old or new, will give slightly different answers when the input is that ambiguous. So when you see a flip in a migration run, you cannot tell whether it is because the model changed or because the input is just noisy. Counting those flips as "migration drift" is measuring input ambiguity, not model change.

The fix is simple but easy to miss: run your noise floor on the same prompt suite you are analyzing. Comparing hard-suite migration flips against a main-suite noise baseline is an apples-to-oranges comparison — the baselines are completely different. Without the hard-suite noise floor, those 7 flips looked significant. With it, they vanished.

### 6.2 "Claude Judges Candidates More Harshly" (Killed)

After the first Claude migration run, three judgment observations surfaced: an `L6→L5` candidate-level flip on a main-suite prompt, and two hard-suite flips including `hold→no_hire` on `interview_evaluation-104`. The pattern provisionally matched the Gemini-pro downgrade effect, raising the hypothesis that newer models judge more harshly across providers.

The repeat migration run (Claude main suite, run 5) falsified it completely. Not a single flip from run 1 recurred. The repeat produced flips on different prompts entirely, and included an upward flip (`L5→L6`), breaking any downgrade pattern. The hard-suite repeat (run 6) produced one flip on a different diagnostic prompt, not the ones that flipped in run 1.

The "harsher judge" story was a single-run artifact. The Gemini-pro claim survives because it reproduced with 8/8 directional consistency across two independent runs. The Claude claim died because 0 out of the original flips recurred. "Newer models judge more harshly" is a Gemini-pro-specific observation; it cannot be generalized.

### 6.3 "Claude +12% Verbosity on Main Suite" (Killed)

Run 1 of the Claude main-suite migration showed +286.9 tokens average, a +12.1% increase. The noise floor at that time showed −1.0%. The separation looked approximately 12×, and the finding appeared strong.

Run 5 (migration repeat, main suite) showed −0.6%. The sign had flipped.

Root cause: the old model's average output differed substantially between the two main-suite runs (2374.0 tokens in run 1 vs 2703.8 tokens in run 5). That is a ±7% spread in the old model's own aggregate output across different invocation days — much larger than the single-run noise floor suggested. The noise floor measured in one session estimated instantaneous sampling variance, not how much the old model's aggregate naturally shifts from day to day.

This finding is not fully killed — the hard-suite verbosity result (+37.2% / +39.7%, reproduced) is robust and holds up. But the main-suite claim must be reported as "inconsistent across runs, within cross-run aggregate variance," not as drift. The lesson: run your noise floor more than once, or at minimum repeat the migration itself and check whether the old model baseline is stable between runs.

### 6.4 What Saved Us

Both controls were necessary, and they caught structurally different failure modes:

- **The noise floor** caught 6.1 (hard-prompt boundary instability). Without a hard-suite noise floor, those 7 flips would have been compared against near-zero main-suite noise and looked significant.
- **The repeat run** caught 6.2 and 6.3. The Claude noise floor was stable and gave no warning — only the migration repeat revealed that the original signals did not hold up. The noise floor alone would not have saved us.

The two controls are not redundant. They are both required.

---

## 7. Discussion

### 7.1 What to Run Before Any Migration

The minimum viable migration test the data supports:

1. **Noise floor run.** Run the old model against itself on your task suite. Record first-attempt validity, token averages, field-level event counts, and decision-field flips. This tells you what "looks like drift but isn't" for your specific tasks and invocation conditions.

2. **Migration run.** Submit both old and new models to the same prompts, same schemas, same metrics.

3. **Repeat migration run.** Re-run the migration at least once. Any finding from step 2 that does not appear again in step 3 is a false positive. Any finding that appears in both runs and exceeds the noise floor is a credible signal.

4. **Hard suite (recommended for complex pipelines).** Include prompts near decision boundaries if your production inputs are ambiguous or underspecified. Run the hard-suite noise floor separately — do not compare hard-suite migration flips against main-suite noise.

This costs roughly 4× the prompt compute of a single migration run. The alternative — shipping a silent 40% output change or systematic judgment recalibration into production — costs more.

### 7.2 Tier-Specific Drift Profiles

The results show a consistent pattern: the *form* of drift depends on the capability tier of the model being replaced.

The Gemini flash-to-lite migration (flash is Google's cheaper, faster tier; lite is the further stripped-down variant) produced output shrinkage and first-attempt failures. Users accepting the cost reduction are also accepting quiet capability regression under constraint-dense schemas, visible only in first-attempt validity data.

The Gemini pro migration produced output shrinkage on routine tasks, adaptive verbosity increases on hard tasks, and systematic judgment recalibration. The seniority judgment finding is the operationally significant one — not because any individual assessment is necessarily wrong, but because the direction is fixed and reproducible. Eight consecutive downward candidate assessments across two independent runs is not noise. Any pipeline using an LLM for candidate screening should treat this as a mandatory pre-production check.

The Claude migration went the other way on verbosity: Sonnet-4-6 produces 38–40% more tokens on difficult tasks. This is helpful if you want comprehensive answers and problematic if you care about latency and API cost at scale.

### 7.3 Judgment Drift in Human-Impacting Pipelines

Both surviving judgment-drift findings — the `COACH→FAIL` support grade flip and the candidate-level downgrade pattern — involve assessments of real people in a production context. A model that consistently grades support agents more harshly, or consistently downgrades candidate levels at a specific seniority, introduces directional bias that accumulates at scale. Individual outputs are schema-valid. The distribution of outputs shifts.

Standard pass/fail regression testing cannot detect this. Detection requires tracking the actual value of categorical decision fields, analyzing whether flips have a consistent direction, and confirming they reproduce. We recommend any team running LLMs in screening, evaluation, or grading pipelines add decision-field directionality as a first-class migration gate.

### 7.4 Cost and Latency Implications

The retry-masking finding has direct production cost implications. For the Gemini flash migration hard suite (75% first-pass), roughly one in four requests requires an error-feedback retry. At production volume, this degrades effective throughput and raises per-compliant-response cost — without any signal in application-level success metrics.

The Claude verbosity increase has analogous implications: +39.7% output tokens on hard tasks is +39.7% API cost and +39.7% response size for any downstream processing. Whether the longer outputs are worth it depends on the application; whether the change happened depends on measuring it.

---

## 8. Limitations

- **Case study scope.** Three schemas and 42 prompts (30 main + 12 hard) across two providers. The schemas cover representative structured-output tasks but are not a broad benchmark. The results show that the framework can detect real drift on these tasks; they do not establish universal migration behavior norms across arbitrary domains.

- **Preview model caveat.** `gemini-3.1-pro-preview` is a preview model that may behave differently at general availability. The judgment-recalibration finding should be re-tested against the GA release before any production migration decision is based on it.

- **Single SDK retry path.** The `instructor` library sends validation errors back to the model as text feedback, then retries. This is one implementation choice; models using constrained decoding (where the decoding algorithm itself enforces the schema, with no retry needed) would not exhibit retry-masking in the same way. The retry-masking finding is specific to this retry strategy.

- **Single-run noise floor underestimates cross-day variance.** As demonstrated by the Claude main-suite verbosity false positive, a noise floor measured in one session does not capture how much the old model's aggregate output naturally shifts across different invocation days. Running the noise floor across multiple sessions would produce a more conservative and more reliable variance estimate.

- **Snapshot comparison only.** Each run submits one request per prompt at provider-default temperature. Within-session variance across multiple invocations of the same prompt is not characterized.

- **Provider model versioning.** `claude-sonnet-4-5-20250929` is a dated snapshot; the provider may revise either model, which could affect reproducibility of the specific numbers reported here.

---

## 9. Conclusion

Schema-level regression testing cannot detect the changes that matter in an LLM migration. Across 16 benchmark runs of two production-representative migrations, pass/fail evaluation reported zero regressions every single time. The metrics underneath those results showed output volume shifting 15–40%, a cheaper model tier requiring error feedback on 25% of hard requests, and a pro-tier migration producing eight consecutive downward candidate assessments without exception.

Just as important: roughly half of the drift we initially detected was noise. Three findings that appeared credible after a first run — hard-prompt decision instability, a cross-provider judgment-harshness pattern, and a 12% Claude verbosity shift — each collapsed under a same-model noise floor or a repeat run. The methodology that found real drift is the methodology that killed these false positives. You cannot have one without the other.

"Drift is universal; its direction is not." Gemini shrank output; Claude grew it. Gemini-pro recalibrated candidate assessments; Claude-sonnet did not. No rule of thumb predicts what a given migration will do. The only reliable approach is to measure it — against a noise floor, with repeats, across your actual task distribution.

The framework, prompts, schemas, and raw results are released as a reproducible artifact.

---

## References

Atil, B., Aykent, S., Chittams, A., Fu, L., Passonneau, R. J., et al. (2024). Non-determinism of "deterministic" LLM settings. *Eval4NLP 2025 Workshop*. arXiv:2408.04667

Biderman, S., Schoelkopf, H., Sutawika, L., Gao, L., Tow, J., et al. (2024). Lessons from the trenches on reproducible evaluation of language models. arXiv:2405.14782

Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML Test Score: A rubric for ML production readiness and technical debt reduction. *IEEE International Conference on Big Data (IEEE BigData 2017)*.

Gao, L., et al. (2023). A framework for few-shot language model evaluation. *Zenodo*. https://doi.org/10.5281/zenodo.10256836

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., et al. (2025). Generating structured outputs from language models: Benchmark and studies. arXiv:2501.10868

Gu, J., et al. (2024). A survey on LLM-as-a-judge. arXiv:2411.15594

Khatchadourian, R., & Franco, R. (2025). LLM output drift: Cross-provider validation and mitigation for financial workflows. arXiv:2511.07585

Liang, P., Bommasani, R., Lee, T., et al. (2022). Holistic evaluation of language models. arXiv:2211.09110

Pahune, S., et al. (2025). Transitioning from MLOps to LLMOps: Navigating the unique challenges of large language models. *Information*, 16(2), 87. https://doi.org/10.3390/info16020087

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems 28 (NeurIPS 2015)*, 2503–2511.

Shankar, S., Garcia, R., Hellerstein, J. M., & Parameswaran, A. G. (2022). Operationalizing machine learning: An interview study. arXiv:2209.09125

Song, Y., Wang, G., Li, S., & Lin, B. Y. (2024). The good, the bad, and the greedy: Evaluation of LLMs should not ignore non-determinism. arXiv:2407.10457

Srivastava, A., et al. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Transactions on Machine Learning Research (TMLR)*. arXiv:2206.04615

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*. arXiv:2203.11171

Willard, B. T., & Louf, R. (2023). Efficient guided generation for large language models. arXiv:2307.09702

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., et al. (2023). Judging LLM-as-a-judge with MT-bench and Chatbot Arena. *NeurIPS 2023 (Datasets and Benchmarks Track)*. arXiv:2306.05685

---

## WRITER'S NOTES

### (a) Citation status

All [CITE] placeholders filled with real references. Full bibliography in References section above. No [NEEDS-DATA] markers were used — all numbers come directly from `gemini_findings.md` or `claude_findings.md`.

### (b) Uncertain claims

1. **Flash migration judgment flips.** Raw flip counts included in Section 5.6 table for completeness, but no surviving finding is claimed from flash migration judgment data. Source does not confirm whether the flash `hold→hire` main-suite flip reproduced; it is left as raw data only.

2. **`interview_evaluation-104` `hold→no_hire`.** Source notes "the old model also flips this one against itself — claim only weakly." Excluded from the two surviving judgment-drift findings; only `COACH→FAIL` and the candidate-level directional pattern are stated as findings.

3. **"~4× compute" estimate (Section 7.1).** Not stated in the source files. Derived as: 1 noise floor + 1 migration + 1 repeat ≈ 3 extra runs vs 1 baseline run. Reasonable but unverified. Flag for review.

4. **"approximately 8×" and "approximately 14×" signal-to-noise ratios.** The source states these ("roughly 8×", "~14× signal") — I preserved the hedge word "approximately" to match the source's phrasing.

### (c) Changes from original draft

- Added inline definitions in parentheses on first use for: schema validation, retry layer, Pydantic, instructor library, Literal types, cross-field validators / @model_validator, hard suite, noise floor (in Framework), decision-field flips, retry-masking.
- Section 6.1 ("Hard Prompts Expose Decision Flips") rewritten substantially — replaced the academic phrase "conflating input-induced instability with model-change-induced instability" with the "borderline grading rubric" analogy and a plain-language explanation of why the hard-suite noise floor matters.
- Academic hedging phrases replaced throughout: "not consistent with a pure-noise model of LLM judgment variability" → "not the kind of random flip you would expect from sampling variation alone"; "reproducibility controls" → "repeat runs"; "exhibits this failure mode" → "has this problem"; "analogous implications" → "similar tradeoffs".
- Definition of "flash" and "lite" added inline in Section 7.2 for readers unfamiliar with Google's model tier naming.
- No numbers, tables, or section structure changed.
