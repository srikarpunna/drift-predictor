from __future__ import annotations

import time
from typing import Type

import instructor
from google import genai
from pydantic import BaseModel as PydanticBaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import RunResult
from src.runners.base_runner import BaseRunner
from src.utils.config import Settings


class GeminiRunner(BaseRunner):
    def __init__(self, model_id: str, settings: Settings) -> None:
        super().__init__(model_id)
        self._retry_cfg = settings.retry
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._instructor_client: instructor.Instructor = instructor.from_genai(
            client=self._client,
            mode=instructor.Mode.GENAI_TOOLS,
        )

    def _make_retry(self):
        return retry(
            stop=stop_after_attempt(self._retry_cfg.max_attempts),
            wait=wait_exponential(
                multiplier=self._retry_cfg.wait_multiplier,
                min=self._retry_cfg.wait_min_seconds,
                max=self._retry_cfg.wait_max_seconds,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    def _run_text_impl(self, prompt: str, prompt_id: str) -> RunResult:
        retrying = self._make_retry()

        @retrying
        def _call():
            return self._client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )

        start = time.monotonic()
        response = _call()
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = response.usage_metadata
        return RunResult(
            model_id=self.model_id,
            prompt_id=prompt_id,
            output_text=response.text or "",
            tokens_in=usage.prompt_token_count if usage else 0,
            tokens_out=usage.candidates_token_count if usage else 0,
            latency_ms=elapsed_ms,
        )

    def _run_structured_impl(
        self,
        prompt: str,
        prompt_id: str,
        schema: Type[PydanticBaseModel],
    ) -> RunResult:
        retrying = self._make_retry()

        @retrying
        def _call():
            return self._instructor_client.chat.completions.create_with_completion(
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                model=self.model_id,
            )

        start = time.monotonic()
        parsed, raw_response = _call()
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = getattr(raw_response, "usage_metadata", None)
        return RunResult(
            model_id=self.model_id,
            prompt_id=prompt_id,
            output_text=parsed.model_dump_json(),
            output_parsed=parsed,
            tokens_in=usage.prompt_token_count if usage else 0,
            tokens_out=usage.candidates_token_count if usage else 0,
            latency_ms=elapsed_ms,
        )
