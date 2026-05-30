"""
pipeline.py
-----------
The main drift detection pipeline.

Orchestrates:
  1. Schema analysis (blast radius)
  2. Model run per transcript
  3. Full recursive schema validation on each output
  4. Report generation

Usage example:
    from pipeline import DriftPipeline

    pipeline = DriftPipeline(
        provider="openai",
        api_key="sk-...",
        model_id="gpt-4o",
        system_prompt=SYSTEM_PROMPT,       # your prod system prompt (string)
        schema=SCHEMA,                     # your prod JSON schema (dict)
        user_prompt_template=TEMPLATE,     # f-string with {transcript} placeholder
    )

    transcripts = {
        "call_001": "<transcript>...",
        "call_002": "<transcript>...",
    }

    report = pipeline.run(transcripts)
    report.print_summary()
    path = report.save_json("./reports")
    print(f"Full report saved → {path}")
"""

from __future__ import annotations
import uuid
from typing import Callable, Optional

from validator import validate, analyze_schema
from model_runner import ModelRunner
from reporter import DriftReport, make_transcript_result


class DriftPipeline:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model_id: str,
        system_prompt: str,
        schema: dict,
        user_prompt_template: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        check_extra_fields: bool = False,
        on_transcript_done: Optional[Callable[[str, bool, int], None]] = None,
    ):
        """
        Parameters
        ----------
        provider              : "openai" | "gemini" | "claude"
        api_key               : API key for the chosen provider
        model_id              : e.g. "gpt-4o", "gemini-1.5-pro", "claude-sonnet-4-6"
        system_prompt         : the full prod system prompt (string)
        schema                : the prod JSON Schema (dict, inline, no $ref)
        user_prompt_template  : string with {transcript} placeholder that gets
                                filled per call. Wrap your full prompt XML here.
        temperature           : model temperature (default 0.0 for determinism)
        max_tokens            : max output tokens per call
        check_extra_fields    : if True, flag fields not in schema as EXTRA_FIELD
        on_transcript_done    : optional callback(transcript_id, passed, violations)
                                called after each transcript completes
        """
        self.runner = ModelRunner(provider=provider, api_key=api_key, model_id=model_id, temperature=temperature)
        self.system_prompt = system_prompt
        self.schema = schema
        self.user_prompt_template = user_prompt_template
        self.max_tokens = max_tokens
        self.check_extra_fields = check_extra_fields
        self.on_transcript_done = on_transcript_done
        self._schema_stats = analyze_schema(schema)

    def run(self, transcripts: dict[str, str], run_id: Optional[str] = None) -> DriftReport:
        """
        Run the pipeline against a set of transcripts.

        Parameters
        ----------
        transcripts : dict of {transcript_id: transcript_text}
        run_id      : optional unique run identifier (auto-generated if None)

        Returns
        -------
        DriftReport with full results
        """
        if run_id is None:
            run_id = uuid.uuid4().hex[:8]

        report = DriftReport(
            run_id=run_id,
            provider=self.runner.provider,
            model_id=self.runner.model_id,
            schema_stats=self._schema_stats,
        )

        total = len(transcripts)
        for i, (tid, transcript_text) in enumerate(transcripts.items(), 1):
            print(f"[{i}/{total}] Running transcript: {tid} ...", end=" ", flush=True)

            # Build user content from template
            user_content = self.user_prompt_template.format(transcript=transcript_text)

            # Call the model
            run_result = self.runner.run(
                system_prompt=self.system_prompt,
                user_content=user_content,
                max_tokens=self.max_tokens,
            )

            # Validate output
            if run_result.parsed_json is not None:
                violations = validate(
                    schema=self.schema,
                    data=run_result.parsed_json,
                    check_extra_fields=self.check_extra_fields,
                )
            else:
                violations = []   # parse failure is captured separately

            transcript_result = make_transcript_result(tid, run_result, violations)
            report.transcript_results.append(transcript_result)

            status = "✅" if transcript_result.passed else ("🔴 parse fail" if transcript_result.parse_failed else f"❌ {len(violations)} violations")
            print(f"{status}  ({run_result.latency_ms:.0f}ms, {run_result.output_tokens} out tokens)")

            if self.on_transcript_done:
                self.on_transcript_done(tid, transcript_result.passed, len(violations))

        return report

    def analyze_schema(self) -> dict:
        """Return a summary of how many conditions exist in your schema."""
        s = self._schema_stats
        return {
            "total_fields": s.total_fields,
            "required_fields": s.required_fields,
            "optional_fields": s.optional_fields,
            "enum_constraints": s.enum_constraints,
            "type_constraints": s.type_constraints,
            "array_schemas": s.array_schemas,
            "max_depth": s.max_depth,
            "all_field_paths": s.field_paths,
        }
