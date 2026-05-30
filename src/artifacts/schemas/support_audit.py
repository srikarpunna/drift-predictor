from typing import Optional
from pydantic import BaseModel, Field, model_validator


class SupportCallAudit(BaseModel):
    """Customer support call audit with compliance checks and quality dimensions."""

    class ComplianceCheck(BaseModel):
        item: str
        passed: str = Field(description="Y, N, or N/A")
        evidence: str = Field(min_length=10, description="Direct evidence from transcript supporting this score. Must be specific quote or observation.")

    class QualityDimension(BaseModel):
        dimension: str
        score: str = Field(description="G, Y, or R")
        evidence: str = Field(min_length=20, description="Specific transcript evidence supporting this score")

    class HoldEvent(BaseModel):
        hold_announced: bool
        hold_duration_seconds: int = Field(ge=0)
        thanked_after_hold: bool

    class AgentAction(BaseModel):
        action_type: str = Field(description="refund_issued, cancellation_processed, account_updated, info_provided, escalation_initiated")
        action_detail: str = Field(min_length=15, description="Specific description of what the agent did")
        authorized: bool = Field(description="True if action was within agent authority; False if it exceeded policy")

    ticket_type: str = Field(description="billing, technical, account, or general")
    agent_identified_correctly: bool
    customer_verified: bool
    compliance_checks: list[ComplianceCheck]
    quality_dimensions: list[QualityDimension]
    resolution_achieved: bool
    escalation_required: bool
    escalation_reason: Optional[str] = Field(default=None, description="Required if escalation_required is true. Null otherwise.")
    overall_grade: str = Field(description="PASS, COACH, or FAIL")
    call_summary: str = Field(min_length=40)
    coaching_notes: Optional[str] = None
    followup_required: bool = Field(description="True if a follow-up action is needed after the call (e.g., pending refund, callback scheduled)")
    followup_reason: Optional[str] = Field(default=None, description="Required if followup_required is true. Null otherwise.")
    hold_event: Optional[HoldEvent] = Field(default=None, description="Populated only if a hold occurred during the call. Null if no hold.")
    agent_actions: list[AgentAction] = Field(default_factory=list, description="All discrete actions the agent took during the call")
    compliance_score_pct: float = Field(ge=0.0, le=100.0, description="Percentage of applicable compliance checks scored Y. Must be < 100.0 if any check scored N.")

    @model_validator(mode='after')
    def escalation_consistency(self):
        if self.escalation_required and not self.escalation_reason:
            raise ValueError("escalation_reason required when escalation_required=True")
        if not self.escalation_required and self.escalation_reason is not None:
            raise ValueError("escalation_reason must be None when escalation_required=False")
        return self

    @model_validator(mode='after')
    def followup_consistency(self):
        if self.followup_required and not self.followup_reason:
            raise ValueError("followup_reason required when followup_required=True")
        if not self.followup_required and self.followup_reason is not None:
            raise ValueError("followup_reason must be None when followup_required=False")
        return self

    @model_validator(mode='after')
    def compliance_score_consistency(self):
        has_failure = any(c.passed == "N" for c in self.compliance_checks)
        if has_failure and self.compliance_score_pct >= 100.0:
            raise ValueError(
                "compliance_score_pct must be < 100.0 when at least one compliance check is scored N"
            )
        return self
