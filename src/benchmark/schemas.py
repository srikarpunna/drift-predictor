"""
Pydantic schemas used in the benchmark.

Each schema is what a production system would define.
Instructor enforces these during run_structured() calls.
Schema validation failure = H2 event (schema_adherence failure).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Classification schemas
# ---------------------------------------------------------------------------

class SentimentOutput(BaseModel):
    label: str = Field(description="One of: positive, negative, neutral")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IntentOutput(BaseModel):
    intent: str = Field(description="One of: question, complaint, compliment, request, other")
    reasoning: Optional[str] = None


class UrgencyOutput(BaseModel):
    urgency: str = Field(description="One of: low, medium, high, critical")
    reason: str


# ---------------------------------------------------------------------------
# Extraction schemas
# ---------------------------------------------------------------------------

class NamedEntities(BaseModel):
    persons: list[str]
    organizations: list[str]
    locations: list[str]


class JobPosting(BaseModel):
    job_title: str
    company: str
    location: str
    salary_mentioned: bool


class ContactInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None


class FinancialFigures(BaseModel):
    revenue_usd_millions: Optional[float] = None
    profit_usd_millions: Optional[float] = None
    yoy_growth_percent: Optional[float] = None


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class ProductListing(BaseModel):
    product_name: str
    price_usd: float
    in_stock: bool
    tags: list[str]


class UserPermissions(BaseModel):
    class UserInfo(BaseModel):
        id: int
        name: str
        email: str
        role: str

    user: UserInfo
    permissions: list[str]


class MeetingNotes(BaseModel):
    meeting_date: str
    attendees: list[str]
    decisions: list[str]
    action_items: list[str]


class ParsedAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None


class CodeAnalysis(BaseModel):
    language: str
    has_bugs: bool
    complexity: str = Field(description="One of: low, medium, high")
    issues: list[str]


# ---------------------------------------------------------------------------
# Transformation schemas
# ---------------------------------------------------------------------------

class CsvToJson(BaseModel):
    name: str
    age: int
    city: str
    occupation: str


class NormalizedPhone(BaseModel):
    normalized: str


class RedactedText(BaseModel):
    redacted_text: str
    pii_count: int


class FormalEmail(BaseModel):
    subject: str
    body: str


class CamelCaseList(BaseModel):
    converted: list[str]


# ---------------------------------------------------------------------------
# Schema registry — name → class
# Used by benchmark_runner to look up schema by PromptItem.schema_name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    # classification
    "SentimentOutput": SentimentOutput,
    "IntentOutput": IntentOutput,
    "UrgencyOutput": UrgencyOutput,
    # extraction
    "NamedEntities": NamedEntities,
    "JobPosting": JobPosting,
    "ContactInfo": ContactInfo,
    "FinancialFigures": FinancialFigures,
    # structured output
    "ProductListing": ProductListing,
    "UserPermissions": UserPermissions,
    "MeetingNotes": MeetingNotes,
    "ParsedAddress": ParsedAddress,
    "CodeAnalysis": CodeAnalysis,
    # transformation
    "CsvToJson": CsvToJson,
    "NormalizedPhone": NormalizedPhone,
    "RedactedText": RedactedText,
    "FormalEmail": FormalEmail,
    "CamelCaseList": CamelCaseList,
}


def get_schema(name: str) -> type[BaseModel]:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(f"Schema '{name}' not in registry. Available: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[name]
