"""
reporter.py
-----------
Generates structured drift reports from validation results.

A DriftReport captures:
  - Per-transcript violation details (field path, type, expected, found)
  - Aggregate counts: total violations, breakdown by type
  - Token / latency metrics per run
  - Schema condition counts (blast radius analysis)
  - JSON parse failures (model produced un-parseable output)

Outputs:
  - Python dict  → for programmatic use / storage
  - Console text → human-readable summary
  - JSON file    → for downstream tooling
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from validator import Violation, MISSING_FIELD, WRONG_TYPE, ENUM_VIOLATION, EXTRA_FIELD, SchemaStats
from model_runner import RunResult


# ── Per-transcript result ─────────────────────────────────────────────────────

@dataclass
class TranscriptResult:
    transcript_id: str
    provider: str
    model_id: str
    parse_failed: bool
    parse_error: Optional[str]
    violations: list[Violation]
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def passed(self) -> bool:
        return not self.parse_failed and len(self.violations) == 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def violations_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts


# ── Full run report ───────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    run_id: str
    provider: str
    model_id: str
    schema_stats: SchemaStats
    transcript_results: list[TranscriptResult] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Summary stats ─────────────────────────────────────────────────────────

    @property
    def total_transcripts(self) -> int:
        return len(self.transcript_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.transcript_results if r.passed)

    @property
    def failed_count(self) -> int:
        return self.total_transcripts - self.passed_count

    @property
    def parse_failure_count(self) -> int:
        return sum(1 for r in self.transcript_results if r.parse_failed)

    @property
    def total_violations(self) -> int:
        return sum(r.violation_count for r in self.transcript_results)

    @property
    def avg_latency_ms(self) -> float:
        times = [r.latency_ms for r in self.transcript_results]
        return sum(times) / len(times) if times else 0.0

    @property
    def avg_output_tokens(self) -> float:
        tokens = [r.output_tokens for r in self.transcript_results]
        return sum(tokens) / len(tokens) if tokens else 0.0

    def violations_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.transcript_results:
            for vtype, cnt in r.violations_by_type().items():
                counts[vtype] = counts.get(vtype, 0) + cnt
        return counts

    def most_broken_paths(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return field paths that violated most frequently across all transcripts."""
        path_counts: dict[str, int] = {}
        for r in self.transcript_results:
            for v in r.violations:
                path_counts[v.path] = path_counts.get(v.path, 0) + 1
        return sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "provider": self.provider,
            "model_id": self.model_id,
            "schema_blast_radius": {
                "total_fields": self.schema_stats.total_fields,
                "required_fields": self.schema_stats.required_fields,
                "optional_fields": self.schema_stats.optional_fields,
                "enum_constraints": self.schema_stats.enum_constraints,
                "type_constraints": self.schema_stats.type_constraints,
                "array_schemas": self.schema_stats.array_schemas,
                "max_depth": self.schema_stats.max_depth,
            },
            "summary": {
                "total_transcripts": self.total_transcripts,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "parse_failures": self.parse_failure_count,
                "total_violations": self.total_violations,
                "violations_by_type": self.violations_by_type(),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "avg_output_tokens": round(self.avg_output_tokens, 1),
            },
            "most_broken_paths": [
                {"path": p, "count": c} for p, c in self.most_broken_paths()
            ],
            "transcripts": [
                {
                    "transcript_id": r.transcript_id,
                    "passed": r.passed,
                    "parse_failed": r.parse_failed,
                    "parse_error": r.parse_error,
                    "violation_count": r.violation_count,
                    "violations_by_type": r.violations_by_type(),
                    "latency_ms": round(r.latency_ms, 1),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "violations": [v.as_dict() for v in r.violations],
                }
                for r in self.transcript_results
            ],
        }

    def save_json(self, output_dir: str = ".") -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"drift_report_{self.run_id}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    # ── Console output ────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        SEP = "─" * 70
        print(f"\n{SEP}")
        print(f"  DRIFT REPORT  |  {self.provider.upper()} / {self.model_id}")
        print(f"  Run ID: {self.run_id}  |  {self.generated_at}")
        print(SEP)

        # Schema blast radius
        s = self.schema_stats
        print(f"\n📐 SCHEMA BLAST RADIUS")
        print(f"   Total fields        : {s.total_fields}")
        print(f"   Required fields     : {s.required_fields}")
        print(f"   Enum constraints    : {s.enum_constraints}")
        print(f"   Type constraints    : {s.type_constraints}")
        print(f"   Array item schemas  : {s.array_schemas}")
        print(f"   Max nesting depth   : {s.max_depth}")

        # Run summary
        print(f"\n📊 RUN SUMMARY ({self.total_transcripts} transcripts)")
        print(f"   ✅ Passed           : {self.passed_count}")
        print(f"   ❌ Failed           : {self.failed_count}")
        print(f"   🔴 Parse failures   : {self.parse_failure_count}")
        print(f"   Total violations    : {self.total_violations}")
        print(f"   Avg latency         : {self.avg_latency_ms:.0f} ms")
        print(f"   Avg output tokens   : {self.avg_output_tokens:.0f}")

        # Violation breakdown
        vbt = self.violations_by_type()
        if vbt:
            print(f"\n🔍 VIOLATIONS BY TYPE")
            for vtype, cnt in sorted(vbt.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * min(cnt, 30)
                print(f"   {vtype:<20} {cnt:>4}  {bar}")

        # Most broken paths
        broken = self.most_broken_paths(10)
        if broken:
            print(f"\n💥 TOP BROKEN FIELD PATHS")
            for path, cnt in broken:
                print(f"   [{cnt:>3}x]  {path}")

        # Per-transcript detail
        print(f"\n📋 PER-TRANSCRIPT DETAIL")
        for r in self.transcript_results:
            status = "✅ PASS" if r.passed else ("🔴 PARSE_FAIL" if r.parse_failed else "❌ FAIL")
            print(f"\n   {status}  [{r.transcript_id}]")
            if r.parse_error:
                print(f"          Parse error: {r.parse_error[:120]}")
            for v in r.violations[:20]:   # cap at 20 per transcript in console
                icon = {"MISSING_FIELD": "⬜", "WRONG_TYPE": "🟡", "ENUM_VIOLATION": "🟠", "EXTRA_FIELD": "🔵"}.get(v.violation_type, "❓")
                print(f"          {icon} {v.violation_type:<20} {v.path}")
                print(f"               expected : {v.expected}")
                print(f"               found    : {v.found}")
            if len(r.violations) > 20:
                print(f"          ... and {len(r.violations) - 20} more (see JSON report)")

        print(f"\n{SEP}\n")


# ── Factory: build a TranscriptResult from a RunResult + violations ───────────

def make_transcript_result(
    transcript_id: str,
    run_result: RunResult,
    violations: list[Violation],
) -> TranscriptResult:
    return TranscriptResult(
        transcript_id=transcript_id,
        provider=run_result.provider,
        model_id=run_result.model_id,
        parse_failed=run_result.parsed_json is None,
        parse_error=run_result.parse_error,
        violations=violations,
        input_tokens=run_result.input_tokens,
        output_tokens=run_result.output_tokens,
        latency_ms=run_result.latency_ms,
    )
