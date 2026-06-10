from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PromptItem(BaseModel):
    """A single benchmark item — one prompt linked to its schema by filename convention."""

    id: str = Field(description="Filename stem, e.g. 'interview_evaluation-001'")
    output_schema: str = Field(description="Schema filename stem, e.g. 'interview_evaluation'")
    prompt_text: str


class BenchmarkRun(BaseModel):
    """Result of running one PromptItem on both old and new models."""

    prompt_id: str
    schema_name: str

    old_model_id: str
    new_model_id: str

    old_output: str
    new_output: str

    old_tokens_in: int
    old_tokens_out: int
    new_tokens_in: int
    new_tokens_out: int

    old_latency_ms: float
    new_latency_ms: float

    old_passed: bool
    new_passed: bool
    old_evidence: Optional[str]
    new_evidence: Optional[str]

    migration_failed: bool    # old passed AND new failed
    migration_improved: bool  # old failed AND new passed

    # Validation attempt tracking — how many schema validation rounds each model needed
    old_first_attempt_valid: bool = True
    new_first_attempt_valid: bool = True
    old_validation_attempts: int = 1
    new_validation_attempts: int = 1

    # First-attempt validation errors (drift signal even when retry rescues the run)
    old_first_attempt_error: Optional[str] = None
    new_first_attempt_error: Optional[str] = None

    # Fine-grained drift: field-level size changes + categorical flags.
    # A run can "pass" while still drifting — these capture what pass/fail misses.
    field_drift_events: list[dict] = Field(default_factory=list)
    # Decision-field value changes (e.g. hire -> no_hire). The highest-stakes signal.
    decision_flips: list[dict] = Field(default_factory=list)
    drift_flags: list[str] = Field(default_factory=list)

    @property
    def token_delta(self) -> int:
        """How many more output tokens new model used vs old."""
        return self.new_tokens_out - self.old_tokens_out

    @property
    def token_delta_pct(self) -> float:
        """Output token change as percent of old model's output."""
        return self.token_delta / max(self.old_tokens_out, 1) * 100

    @property
    def drifted(self) -> bool:
        return bool(self.drift_flags)
