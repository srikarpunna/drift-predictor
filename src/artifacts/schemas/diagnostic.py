from pydantic import BaseModel, Field, model_validator


class DiagnosticReport(BaseModel):
    """QA diagnostic report with metrics, failure patterns, and recommendations."""

    class MetricSummary(BaseModel):
        metric_name: str
        value: str
        interpretation: str = Field(min_length=20)

    class FailurePattern(BaseModel):
        pattern_name: str
        frequency: str = Field(description="e.g. '3/10 runs', '30%'")
        example_run_ids: list[str] = Field(description="Specific run IDs from the data that exemplify this pattern")
        root_cause: str = Field(min_length=35)
        recommendation: str = Field(min_length=35)

    class FamilyBreakdown(BaseModel):
        task_family: str
        regression_count: int = Field(ge=0)
        total_count: int = Field(ge=1)
        rate_pct: float = Field(ge=0.0, le=100.0, description="regression_count / total_count * 100, within 1% tolerance")

    executive_summary: str = Field(min_length=80)
    metrics: list[MetricSummary]
    failure_patterns: list[FailurePattern]
    consistency_vs_accuracy_note: str = Field(min_length=50)
    priority_fixes: list[str]
    recommended_action: str = Field(description="proceed_migration, block_migration, or conditional_proceed")
    regression_rate_pct: float = Field(ge=0.0, le=100.0, description="Overall regression rate: regressions / previously-passing runs * 100")
    regression_by_family: list[FamilyBreakdown] = Field(description="Per-task-family regression breakdown. Must not be empty when failure_patterns is non-empty.")

    @model_validator(mode='after')
    def priority_fixes_sufficient(self):
        if self.failure_patterns and len(self.priority_fixes) < 2:
            raise ValueError(
                f"priority_fixes must have at least 2 items when failure_patterns is non-empty "
                f"(got {len(self.priority_fixes)})"
            )
        return self

    @model_validator(mode='after')
    def recommended_action_consistent(self):
        valid = {"proceed_migration", "block_migration", "conditional_proceed"}
        if self.recommended_action not in valid:
            raise ValueError(f"recommended_action must be one of {valid}, got '{self.recommended_action}'")
        if self.regression_rate_pct > 50.0 and self.recommended_action == "proceed_migration":
            raise ValueError(
                f"recommended_action cannot be 'proceed_migration' when regression_rate_pct={self.regression_rate_pct} > 50.0"
            )
        return self

    @model_validator(mode='after')
    def regression_by_family_required(self):
        if self.failure_patterns and not self.regression_by_family:
            raise ValueError("regression_by_family must not be empty when failure_patterns is non-empty")
        return self

    @model_validator(mode='after')
    def family_rate_pct_accurate(self):
        for fb in self.regression_by_family:
            expected = (fb.regression_count / fb.total_count) * 100
            if abs(fb.rate_pct - expected) > 1.0:
                raise ValueError(
                    f"FamilyBreakdown({fb.task_family}): rate_pct={fb.rate_pct} does not match "
                    f"regression_count/total_count*100={expected:.1f} (tolerance ±1%)"
                )
        return self
