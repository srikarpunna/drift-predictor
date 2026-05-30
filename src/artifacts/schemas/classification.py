from typing import Optional
from pydantic import BaseModel, Field


class SentimentOutput(BaseModel):
    label: str = Field(description="One of: positive, negative, neutral")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IntentOutput(BaseModel):
    intent: str = Field(description="One of: question, complaint, compliment, request, other")
    reasoning: Optional[str] = None


class UrgencyOutput(BaseModel):
    urgency: str = Field(description="One of: low, medium, high, critical")
    reason: str
