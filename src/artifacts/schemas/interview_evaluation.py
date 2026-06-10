from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class InterviewEvaluationReport(BaseModel):
    """HR interview evaluation with nested checkpoint and quality area structures."""

    class CheckpointResult(BaseModel):
        checkpoint_id: str
        score: Literal["Y", "N", "N/A"]
        excerpt: str = Field(
            description="[HH:MM] CandidateName: exact quote from transcript. Must be empty string \"\" if score is N/A or no direct quote supports the score."
        )
        reasoning: str = Field(min_length=25)

    class QualityArea(BaseModel):
        area: str
        description: str = Field(description="What this area evaluates and why it matters")
        rating: Literal["G", "Y", "R"] = Field(description="G (good, meets bar), Y (yellow, needs coaching), R (red, below bar)")
        reasoning: str = Field(min_length=30)

    class CompetencyScore(BaseModel):
        competency: Literal["system_design", "behavioral_depth", "cross_functional", "communication", "technical_accuracy"]
        score: int = Field(ge=1, le=5, description="1=far below bar, 2=below bar, 3=meets bar, 4=above bar, 5=exceptional")
        rationale: str = Field(min_length=25, description="Specific transcript evidence supporting this score")

    interview_type: Literal["behavioral", "technical", "final"]
    checkpoints: list[CheckpointResult]
    quality_areas: list[QualityArea]
    overall_recommendation: Literal["hire", "no_hire", "hold"]
    panel_recommendation: Literal["advance", "decline", "defer"] = Field(description="Must be consistent with overall_recommendation")
    candidate_level_assessed: Literal["L4", "L5", "L6", "L7", "unknown"] = Field(description="Level demonstrated during the interview")
    competency_scores: list[CompetencyScore] = Field(description="At least one competency score required. Score each dimension the interview covered.")
    key_strengths: list[str]
    key_concerns: list[str]
    risk_flags: list[str] = Field(description="Specific risk signals observed. Empty list if none.")
    summary: str = Field(min_length=50)

    @model_validator(mode='after')
    def recommendation_rules(self):
        panel_map = {"hire": "advance", "no_hire": "decline", "hold": "defer"}
        expected_panel = panel_map.get(self.overall_recommendation)
        if expected_panel and self.panel_recommendation != expected_panel:
            raise ValueError(
                f"panel_recommendation must be '{expected_panel}' when overall_recommendation='{self.overall_recommendation}', "
                f"got '{self.panel_recommendation}'"
            )

        for cp in self.checkpoints:
            if cp.checkpoint_id == "C-01" and cp.score == "N":
                if self.overall_recommendation != "no_hire":
                    raise ValueError(
                        "Auto-downgrade violated: C-01=N requires overall_recommendation='no_hire', "
                        f"got '{self.overall_recommendation}'"
                    )

        if not self.competency_scores:
            raise ValueError("competency_scores must contain at least one entry")

        return self
