"""
Integration smoke tests for GeminiRunner and ClaudeRunner.

Hits real APIs. Requires .env with GEMINI_API_KEY and ANTHROPIC_API_KEY.

Run:
    pytest tests/test_runners.py -v -s -m integration

Rate-limited? Test one model at a time:
    pytest -k "old" -m integration -v -s
"""

from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field

from src.models import RunResult
from src.runners.claude_runner import ClaudeRunner
from src.runners.gemini_runner import GeminiRunner
from src.utils.config import ClaudeModels, GeminiModels, settings


class PersonExtraction(BaseModel):
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age in years if mentioned")
    occupation: Optional[str] = Field(default=None, description="Job or role if mentioned")


@pytest.fixture(scope="module")
def gemini_old() -> GeminiRunner:
    return GeminiRunner(model_id=GeminiModels.OLD, settings=settings)


@pytest.fixture(scope="module")
def gemini_new() -> GeminiRunner:
    return GeminiRunner(model_id=GeminiModels.NEW, settings=settings)


@pytest.fixture(scope="module")
def claude_old() -> ClaudeRunner:
    return ClaudeRunner(model_id=ClaudeModels.OLD, settings=settings)


@pytest.fixture(scope="module")
def claude_new() -> ClaudeRunner:
    return ClaudeRunner(model_id=ClaudeModels.NEW, settings=settings)


SIMPLE_PROMPT = "What is the capital of France? Answer in one sentence."
EXTRACT_PROMPT = (
    "Extract information about this person: "
    "Alice Johnson is a 34-year-old software engineer at a tech startup."
)
PROMPT_SIMPLE = "smoke-simple-v1"
PROMPT_EXTRACT = "smoke-extract-v1"


def assert_valid(result: RunResult, expect_parsed: bool = False) -> None:
    assert result.error is None, f"Run failed: {result.error}"
    assert result.output_text, "output_text empty on success"
    assert result.latency_ms > 0
    assert result.tokens_in >= 0
    assert result.tokens_out >= 0
    if expect_parsed:
        assert isinstance(result.output_parsed, PersonExtraction), (
            f"expected PersonExtraction, got {type(result.output_parsed)}"
        )


@pytest.mark.integration
def test_gemini_old_text(gemini_old: GeminiRunner) -> None:
    result = gemini_old.run_text(SIMPLE_PROMPT, PROMPT_SIMPLE)
    assert_valid(result)
    print(f"\n[{result.model_id}] {result.output_text[:120]}")
    print(f"  tokens {result.tokens_in}in/{result.tokens_out}out  {result.latency_ms:.0f}ms")


@pytest.mark.integration
def test_gemini_new_text(gemini_new: GeminiRunner) -> None:
    result = gemini_new.run_text(SIMPLE_PROMPT, PROMPT_SIMPLE)
    assert_valid(result)
    print(f"\n[{result.model_id}] {result.output_text[:120]}")
    print(f"  tokens {result.tokens_in}in/{result.tokens_out}out  {result.latency_ms:.0f}ms")


@pytest.mark.integration
def test_gemini_old_structured(gemini_old: GeminiRunner) -> None:
    result = gemini_old.run_structured(EXTRACT_PROMPT, PROMPT_EXTRACT, PersonExtraction)
    assert_valid(result, expect_parsed=True)
    extracted: PersonExtraction = result.output_parsed  # type: ignore[assignment]
    assert "alice" in extracted.name.lower() or "johnson" in extracted.name.lower()
    print(f"\n[{result.model_id}] parsed: {extracted}")


@pytest.mark.integration
def test_gemini_new_structured(gemini_new: GeminiRunner) -> None:
    result = gemini_new.run_structured(EXTRACT_PROMPT, PROMPT_EXTRACT, PersonExtraction)
    assert_valid(result, expect_parsed=True)
    extracted: PersonExtraction = result.output_parsed  # type: ignore[assignment]
    assert "alice" in extracted.name.lower() or "johnson" in extracted.name.lower()
    print(f"\n[{result.model_id}] parsed: {extracted}")


@pytest.mark.integration
def test_claude_old_text(claude_old: ClaudeRunner) -> None:
    result = claude_old.run_text(SIMPLE_PROMPT, PROMPT_SIMPLE)
    assert_valid(result)
    print(f"\n[{result.model_id}] {result.output_text[:120]}")
    print(f"  tokens {result.tokens_in}in/{result.tokens_out}out  {result.latency_ms:.0f}ms")


@pytest.mark.integration
def test_claude_new_text(claude_new: ClaudeRunner) -> None:
    result = claude_new.run_text(SIMPLE_PROMPT, PROMPT_SIMPLE)
    assert_valid(result)
    print(f"\n[{result.model_id}] {result.output_text[:120]}")
    print(f"  tokens {result.tokens_in}in/{result.tokens_out}out  {result.latency_ms:.0f}ms")


@pytest.mark.integration
def test_claude_old_structured(claude_old: ClaudeRunner) -> None:
    result = claude_old.run_structured(EXTRACT_PROMPT, PROMPT_EXTRACT, PersonExtraction)
    assert_valid(result, expect_parsed=True)
    extracted: PersonExtraction = result.output_parsed  # type: ignore[assignment]
    assert "alice" in extracted.name.lower() or "johnson" in extracted.name.lower()
    print(f"\n[{result.model_id}] parsed: {extracted}")


@pytest.mark.integration
def test_claude_new_structured(claude_new: ClaudeRunner) -> None:
    result = claude_new.run_structured(EXTRACT_PROMPT, PROMPT_EXTRACT, PersonExtraction)
    assert_valid(result, expect_parsed=True)
    extracted: PersonExtraction = result.output_parsed  # type: ignore[assignment]
    assert "alice" in extracted.name.lower() or "johnson" in extracted.name.lower()
    print(f"\n[{result.model_id}] parsed: {extracted}")


@pytest.mark.integration
def test_run_dispatch_symmetry() -> None:
    """Verify .run(schema=None) → text mode; .run(schema=Schema) → structured mode."""
    gemini = GeminiRunner(model_id=GeminiModels.OLD, settings=settings)
    claude = ClaudeRunner(model_id=ClaudeModels.OLD, settings=settings)

    for runner in [gemini, claude]:
        text_result = runner.run(SIMPLE_PROMPT, PROMPT_SIMPLE, schema=None)
        assert text_result.output_parsed is None, (
            f"{runner.model_id}: text mode must not set output_parsed"
        )

        struct_result = runner.run(EXTRACT_PROMPT, PROMPT_EXTRACT, schema=PersonExtraction)
        assert isinstance(struct_result.output_parsed, PersonExtraction), (
            f"{runner.model_id}: structured mode must set output_parsed"
        )
