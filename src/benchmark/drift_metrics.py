"""Fine-grained drift metrics between old/new model outputs.

Pass/fail alone is too blunt: a model can shrink fields by 60%, drop list
items, and still parse cleanly. These metrics quantify drift that the
schema oracle cannot see.
"""

from __future__ import annotations

import json
from typing import Any

# A string field shrinking/growing by more than this fraction counts as drift.
STRING_DRIFT_THRESHOLD = 0.4
# A per-prompt output token change beyond this percent gets a verbosity flag.
VERBOSITY_FLAG_PCT = 25.0


def field_metrics(obj: Any, prefix: str = "") -> dict[str, int]:
    """Flatten a parsed JSON object into {field_path: size} metrics.

    For each list: '<path>.__len' = item count.
    For each string: '<path>.__chars' = total chars (summed across list items).
    """
    out: dict[str, int] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            for k2, v2 in field_metrics(v, f"{prefix}{k}.").items():
                out[k2] = out.get(k2, 0) + v2
    elif isinstance(obj, list):
        out[prefix + "__len"] = out.get(prefix + "__len", 0) + len(obj)
        for v in obj:
            for k2, v2 in field_metrics(v, prefix).items():
                out[k2] = out.get(k2, 0) + v2
    elif isinstance(obj, str):
        out[prefix + "__chars"] = out.get(prefix + "__chars", 0) + len(obj)
    return out


def compare_outputs(old_output: str, new_output: str) -> list[dict]:
    """Field-level drift events between two JSON output strings.

    Returns a list of {"field", "old", "new"} dicts for:
    - any list whose length changed
    - any string field whose total length changed by > STRING_DRIFT_THRESHOLD
    Returns [] if either output is not parseable JSON.
    """
    try:
        old_metrics = field_metrics(json.loads(old_output))
        new_metrics = field_metrics(json.loads(new_output))
    except (json.JSONDecodeError, TypeError):
        return []

    events = []
    for key in sorted(set(old_metrics) | set(new_metrics)):
        old_v, new_v = old_metrics.get(key, 0), new_metrics.get(key, 0)
        if key.endswith("__len") and old_v != new_v:
            events.append({"field": key, "old": old_v, "new": new_v})
        elif key.endswith("__chars") and old_v > 0:
            if abs(new_v - old_v) / old_v > STRING_DRIFT_THRESHOLD:
                events.append({"field": key, "old": old_v, "new": new_v})
    return events


def drift_flags(
    token_delta_pct: float,
    field_events: list[dict],
    new_first_attempt_valid: bool,
) -> list[str]:
    """Categorical drift flags for one prompt run."""
    flags = []
    if not new_first_attempt_valid:
        flags.append("first_attempt_fail")
    if token_delta_pct <= -VERBOSITY_FLAG_PCT:
        flags.append("verbosity_shrink")
    elif token_delta_pct >= VERBOSITY_FLAG_PCT:
        flags.append("verbosity_grow")
    if any(e["field"].endswith("__len") for e in field_events):
        flags.append("structure_drift")
    if any(
        e["field"].endswith("__chars") and e["new"] < e["old"] for e in field_events
    ):
        flags.append("content_shrink")
    return flags
