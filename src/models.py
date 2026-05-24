from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class RunResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model_id: str
    prompt_id: str

    output_text: str
    output_parsed: Optional[Any] = None

    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)

    error: Optional[str] = None

    @model_validator(mode="after")
    def check_output_or_error(self) -> "RunResult":
        if self.error is None and not self.output_text:
            raise ValueError("output_text must be non-empty when error is None")
        return self

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class MigrationResult(BaseModel):
    old_result: RunResult
    new_result: RunResult

    old_passed: bool
    new_passed: bool

    # True when old passed but new failed — the regression pattern we study
    migration_failed: bool
    failure_oracle: Optional[str] = None
    failure_evidence: Optional[str] = None

    @model_validator(mode="after")
    def validate_migration_failed(self) -> "MigrationResult":
        expected = self.old_passed and not self.new_passed
        if self.migration_failed != expected:
            raise ValueError(
                f"migration_failed={self.migration_failed} inconsistent with "
                f"old_passed={self.old_passed}, new_passed={self.new_passed}"
            )
        return self

    @property
    def is_regression(self) -> bool:
        return self.migration_failed

    @property
    def is_improvement(self) -> bool:
        return not self.old_passed and self.new_passed

    @property
    def latency_delta_ms(self) -> float:
        return self.new_result.latency_ms - self.old_result.latency_ms
