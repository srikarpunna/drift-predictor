"""
validator.py
------------
Recursively walks a JSON Schema and validates an actual model output against it.
Reports every violation — missing fields, wrong types, enum mismatches, deep nesting breaks.

Supports:
  - type: object, array, string, integer, number, boolean, null
  - required fields at every nesting level
  - enum constraints
  - nullable types: ["array", "null"], ["string", "null"], etc.
  - array item schemas (validates every element)
  - $ref is NOT supported (inline schemas only)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── Violation types ───────────────────────────────────────────────────────────

MISSING_FIELD   = "MISSING_FIELD"    # required field absent
WRONG_TYPE      = "WRONG_TYPE"       # value present but wrong type
ENUM_VIOLATION  = "ENUM_VIOLATION"   # value not in allowed enum list
EXTRA_FIELD     = "EXTRA_FIELD"      # field not defined in schema (optional check)


@dataclass
class Violation:
    path: str          # dot-notation path, e.g. "sales_outcome.intend_to_proceed.value"
    violation_type: str
    expected: str      # human-readable expectation
    found: str         # human-readable actual value
    severity: str = "ERROR"   # ERROR | WARNING

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "violation_type": self.violation_type,
            "expected": self.expected,
            "found": self.found,
            "severity": self.severity,
        }


# ── Core recursive validator ──────────────────────────────────────────────────

def _resolve_types(schema: dict) -> list[str]:
    """Return list of allowed types from a schema node."""
    t = schema.get("type")
    if t is None:
        return []
    if isinstance(t, list):
        return t
    return [t]


def _python_type_matches(value: Any, allowed_types: list[str]) -> bool:
    """Check if a Python value matches any of the JSON Schema type names."""
    type_map = {
        "object":  dict,
        "array":   list,
        "string":  str,
        "integer": int,
        "number":  (int, float),
        "boolean": bool,
        "null":    type(None),
    }
    for t in allowed_types:
        py_type = type_map.get(t)
        if py_type is None:
            continue
        if isinstance(value, py_type):
            # JSON Schema: integer must not match float (unless number)
            if t == "integer" and isinstance(value, float):
                continue
            return True
    return False


def _friendly(value: Any) -> str:
    """Short human-readable representation of a value."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return f'"{value}"' if len(value) <= 60 else f'"{value[:57]}..."'
    if isinstance(value, (dict, list)):
        return type(value).__name__
    return str(value)


def validate(
    schema: dict,
    data: Any,
    path: str = "",
    violations: list[Violation] | None = None,
    check_extra_fields: bool = False,
) -> list[Violation]:
    """
    Recursively validate `data` against `schema`.

    Parameters
    ----------
    schema              : JSON Schema dict (inline, no $ref)
    data                : parsed JSON value to validate
    path                : current dot-path (used for reporting)
    violations          : accumulator list (created if None)
    check_extra_fields  : if True, flag fields not defined in the schema

    Returns
    -------
    List of Violation objects, one per problem found.
    """
    if violations is None:
        violations = []

    allowed_types = _resolve_types(schema)

    # ── null shortcut ─────────────────────────────────────────────────────────
    if data is None:
        if "null" in allowed_types or not allowed_types:
            return violations  # null is explicitly allowed or type not constrained
        violations.append(Violation(
            path=path or "<root>",
            violation_type=WRONG_TYPE,
            expected=" | ".join(allowed_types),
            found="null",
        ))
        return violations

    # ── type check ────────────────────────────────────────────────────────────
    if allowed_types and not _python_type_matches(data, allowed_types):
        violations.append(Violation(
            path=path or "<root>",
            violation_type=WRONG_TYPE,
            expected=" | ".join(allowed_types),
            found=type(data).__name__,
        ))
        # Still try to recurse for objects/arrays so we catch MORE violations
        if not isinstance(data, (dict, list)):
            return violations

    # ── enum check ────────────────────────────────────────────────────────────
    if "enum" in schema and data not in schema["enum"]:
        violations.append(Violation(
            path=path or "<root>",
            violation_type=ENUM_VIOLATION,
            expected="one of " + str(schema["enum"]),
            found=_friendly(data),
        ))

    # ── object: recurse into properties ───────────────────────────────────────
    if isinstance(data, dict) and "properties" in schema:
        props: dict = schema["properties"]
        required: list = schema.get("required", [])

        # Check required fields
        for req_field in required:
            child_path = f"{path}.{req_field}" if path else req_field
            if req_field not in data:
                violations.append(Violation(
                    path=child_path,
                    violation_type=MISSING_FIELD,
                    expected="required field to be present",
                    found="<absent>",
                ))
            else:
                child_schema = props.get(req_field, {})
                validate(child_schema, data[req_field], child_path, violations, check_extra_fields)

        # Check non-required fields that ARE present (still validate their shape)
        for present_field, present_value in data.items():
            if present_field in required:
                continue  # already handled above
            child_path = f"{path}.{present_field}" if path else present_field
            if present_field in props:
                validate(props[present_field], present_value, child_path, violations, check_extra_fields)
            elif check_extra_fields:
                violations.append(Violation(
                    path=child_path,
                    violation_type=EXTRA_FIELD,
                    expected="field not defined in schema",
                    found=_friendly(present_value),
                    severity="WARNING",
                ))

    # ── array: recurse into each item ─────────────────────────────────────────
    if isinstance(data, list) and "items" in schema:
        item_schema = schema["items"]
        for idx, item in enumerate(data):
            item_path = f"{path}[{idx}]"
            validate(item_schema, item, item_path, violations, check_extra_fields)

    return violations


# ── Schema condition counter ──────────────────────────────────────────────────

@dataclass
class SchemaStats:
    total_fields: int = 0          # every named property (at any depth)
    required_fields: int = 0       # fields marked required
    optional_fields: int = 0       # fields present but not required
    enum_constraints: int = 0      # fields with enum validation
    type_constraints: int = 0      # fields with explicit type
    array_schemas: int = 0         # arrays with item schemas
    max_depth: int = 0
    field_paths: list[str] = field(default_factory=list)


def analyze_schema(schema: dict, path: str = "", depth: int = 0, stats: SchemaStats | None = None) -> SchemaStats:
    """
    Walk every node in a JSON Schema and count all conditions/checks.
    This tells you the 'blast radius' — how many things can break on a model switch.
    """
    if stats is None:
        stats = SchemaStats()

    stats.max_depth = max(stats.max_depth, depth)

    if "enum" in schema:
        stats.enum_constraints += 1

    if "type" in schema:
        stats.type_constraints += 1

    if isinstance(schema.get("type"), str) and schema["type"] == "array" and "items" in schema:
        stats.array_schemas += 1
        analyze_schema(schema["items"], f"{path}[]", depth + 1, stats)

    if "properties" in schema:
        required = set(schema.get("required", []))
        for prop_name, prop_schema in schema["properties"].items():
            child_path = f"{path}.{prop_name}" if path else prop_name
            stats.total_fields += 1
            stats.field_paths.append(child_path)
            if prop_name in required:
                stats.required_fields += 1
            else:
                stats.optional_fields += 1
            analyze_schema(prop_schema, child_path, depth + 1, stats)

    return stats
