"""
model_runner.py
---------------
Sends a prompt + transcript to a chosen model and returns the raw text response.

Supported providers:
  - openai   → requires openai>=1.0.0
  - gemini   → requires google-generativeai>=0.5.0
  - claude   → requires anthropic>=0.25.0

Usage:
    runner = ModelRunner(provider="openai", api_key="sk-...", model_id="gpt-4o")
    raw = runner.run(system_prompt=..., user_content=...)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Optional


SUPPORTED_PROVIDERS = ("openai", "gemini", "claude")


@dataclass
class RunResult:
    raw_text: str
    parsed_json: Optional[dict]
    parse_error: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    provider: str


class ModelRunner:
    def __init__(self, provider: str, api_key: str, model_id: str, temperature: float = 0.0):
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {SUPPORTED_PROVIDERS}")
        self.provider = provider
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, system_prompt: str, user_content: str, max_tokens: int = 8192) -> RunResult:
        t0 = time.perf_counter()
        if self.provider == "openai":
            result = self._run_openai(system_prompt, user_content, max_tokens)
        elif self.provider == "gemini":
            result = self._run_gemini(system_prompt, user_content, max_tokens)
        elif self.provider == "claude":
            result = self._run_claude(system_prompt, user_content, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _run_openai(self, system_prompt: str, user_content: str, max_tokens: int) -> RunResult:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        return self._make_result(
            raw=raw,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _run_gemini(self, system_prompt: str, user_content: str, max_tokens: int) -> RunResult:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(user_content)
        raw = response.text or ""
        usage = response.usage_metadata
        return self._make_result(
            raw=raw,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
        )

    # ── Claude ────────────────────────────────────────────────────────────────

    def _run_claude(self, system_prompt: str, user_content: str, max_tokens: int) -> RunResult:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text if response.content else ""
        usage = response.usage
        return self._make_result(
            raw=raw,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    def _make_result(self, raw: str, input_tokens: int, output_tokens: int) -> RunResult:
        parsed = None
        parse_error = None
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            parse_error = str(e)

        return RunResult(
            raw_text=raw,
            parsed_json=parsed,
            parse_error=parse_error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,   # filled in by caller
            model_id=self.model_id,
            provider=self.provider,
        )
