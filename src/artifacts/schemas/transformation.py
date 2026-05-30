from pydantic import BaseModel


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
