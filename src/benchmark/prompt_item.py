from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class OracleConfig(BaseModel):
    """Configuration telling the oracle how to evaluate a RunResult."""

    type: Literal["parser", "unit_test", "structured", "rubric", "human"]

    # parser oracle: validate output is valid JSON and optionally matches schema
    required_keys: Optional[list[str]] = None
    forbidden_keys: Optional[list[str]] = None
    schema_def: Optional[dict[str, Any]] = None  # JSON Schema dict

    # unit_test oracle: check specific field values
    # List of {"field": "key.path", "op": "eq|contains|isinstance", "value": ...}
    assertions: Optional[list[dict[str, Any]]] = None

    # rubric oracle: criteria for LLM judge (implemented later)
    rubric: Optional[str] = None

    # shared: max output length (tokens or chars) — None means no limit
    max_output_chars: Optional[int] = None


class PromptItem(BaseModel):
    """A single benchmark item — one prompt with its oracle configuration."""

    id: str = Field(description="Stable unique ID, e.g. 'classification-001'")
    task_family: Literal[
        "classification",
        "extraction",
        "structured_output",
        "tool_calling",
        "summarization",
        "transformation",
        "multi_constraint",
    ]
    prompt_text: str
    oracle: OracleConfig

    # When set, benchmark_runner calls run_structured() with this schema
    # (Instructor path). Name must exist in SCHEMA_REGISTRY.
    # When None, calls run_text() and oracle validates raw text output.
    schema_name: Optional[str] = None

    # Optional: expected output for documentation/reference (not used by oracle)
    example_output: Optional[str] = None


class BenchmarkRun(BaseModel):
    """Result of running one PromptItem on both old and new models."""

    prompt_id: str
    task_family: str

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
    old_evidence: Optional[str]  # failure evidence if failed
    new_evidence: Optional[str]

    migration_failed: bool  # old passed AND new failed
    migration_improved: bool  # old failed AND new passed

    @property
    def token_delta(self) -> int:
        """How many more output tokens new model used vs old."""
        return self.new_tokens_out - self.old_tokens_out
