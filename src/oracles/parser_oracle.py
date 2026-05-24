from __future__ import annotations

import json
import re

from src.benchmark.prompt_item import OracleConfig
from src.models import RunResult
from src.oracles.base_oracle import BaseOracle


class ParserOracle(BaseOracle):
    """
    Validates that model output is parseable JSON and satisfies structural constraints.

    Checks (in order):
    1. Output is valid JSON
    2. All required_keys are present at the top level
    3. No forbidden_keys are present
    4. max_output_chars not exceeded (if set)
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def check(self, result: RunResult) -> tuple[bool, str | None]:
        if not result.succeeded:
            return False, f"run error: {result.error}"

        text = result.output_text.strip()

        # Strip markdown code fences if model wraps JSON in ```json ... ```
        text = _strip_code_fence(text)

        needs_json = bool(
            self.config.required_keys
            or self.config.forbidden_keys
            or self.config.schema_def
        )

        if needs_json:
            # 1. JSON validity
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                return False, f"invalid JSON: {e}"

            if not isinstance(parsed, dict):
                return False, f"expected JSON object, got {type(parsed).__name__}"

            # 2. Required keys
            if self.config.required_keys:
                for key in self.config.required_keys:
                    if key not in parsed:
                        return False, f"missing required key: '{key}'"

            # 3. Forbidden keys
            if self.config.forbidden_keys:
                for key in self.config.forbidden_keys:
                    if key in parsed:
                        return False, f"forbidden key present: '{key}'"

        # 4. Length limit (applies regardless of JSON mode)
        if self.config.max_output_chars and len(result.output_text) > self.config.max_output_chars:
            return False, (
                f"output length {len(result.output_text)} exceeds limit {self.config.max_output_chars}"
            )

        return True, None


class StructuredOracle(BaseOracle):
    """
    Oracle for prompts run through Instructor (run_structured path).

    Pass/fail is determined by whether Instructor successfully parsed the
    Pydantic instance. If result.error is set, Instructor failed = schema
    adherence failure (H2 event).

    Additional assertions run against output_parsed fields using the same
    op system as UnitTestOracle, but operating on the Pydantic model dict.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def check(self, result: RunResult) -> tuple[bool, str | None]:
        # Instructor failure = schema adherence failure
        if not result.succeeded:
            return False, f"schema_adherence_failure: {result.error}"

        if result.output_parsed is None:
            return False, "output_parsed is None despite no error — unexpected state"

        # Run additional field assertions if specified
        if self.config.assertions:
            parsed_dict = result.output_parsed.model_dump()
            for assertion in self.config.assertions:
                field = assertion["field"]
                op = assertion["op"]
                expected = assertion.get("value")
                actual = _get_nested(parsed_dict, field)
                passed, evidence = _apply_op(op, actual, expected, field)
                if not passed:
                    return False, evidence

        return True, None


class UnitTestOracle(BaseOracle):
    """
    Checks specific field values in parsed JSON output using assertion rules.

    Each assertion: {"field": "key", "op": "eq|contains|in|not_null|isinstance", "value": ...}
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def check(self, result: RunResult) -> tuple[bool, str | None]:
        if not result.succeeded:
            return False, f"run error: {result.error}"

        text = _strip_code_fence(result.output_text.strip())

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return False, f"invalid JSON (cannot run assertions): {e}"

        if not self.config.assertions:
            return True, None

        for assertion in self.config.assertions:
            field = assertion["field"]
            op = assertion["op"]
            expected = assertion.get("value")

            # Navigate dotted field paths: "person.name" → parsed["person"]["name"]
            actual = _get_nested(parsed, field)

            passed, evidence = _apply_op(op, actual, expected, field)
            if not passed:
                return False, evidence

        return True, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text


def _get_nested(obj: dict, path: str):
    """Navigate 'a.b.c' paths into nested dicts. Returns None if missing."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _apply_op(
    op: str, actual, expected, field: str
) -> tuple[bool, str | None]:
    if op == "eq":
        if actual != expected:
            return False, f"field '{field}': expected {expected!r}, got {actual!r}"
    elif op == "contains":
        if not isinstance(actual, str) or expected.lower() not in actual.lower():
            return False, f"field '{field}': expected to contain {expected!r}, got {actual!r}"
    elif op == "in":
        if actual not in expected:
            return False, f"field '{field}': {actual!r} not in allowed values {expected}"
    elif op == "not_null":
        if actual is None:
            return False, f"field '{field}': expected non-null value"
    elif op == "isinstance":
        type_map = {"str": str, "int": int, "float": float, "list": list, "dict": dict, "bool": bool}
        expected_type = type_map.get(expected)
        if expected_type and not isinstance(actual, expected_type):
            return False, f"field '{field}': expected type {expected}, got {type(actual).__name__}"
    elif op == "min_length":
        if not isinstance(actual, str) or len(actual) < int(expected):
            return False, f"field '{field}': length {len(actual) if actual else 0} < min {expected}"
    elif op == "max_length":
        if isinstance(actual, str) and len(actual) > int(expected):
            return False, f"field '{field}': length {len(actual)} > max {expected}"
    return True, None
