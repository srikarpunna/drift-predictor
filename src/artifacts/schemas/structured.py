from typing import Optional
from pydantic import BaseModel, Field


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
