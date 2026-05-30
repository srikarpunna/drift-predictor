from typing import Optional
from pydantic import BaseModel


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
