from __future__ import annotations

import time
from typing import Type

import anthropic
import instructor
from pydantic import BaseModel as PydanticBaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import RunResult
from src.runners.base_runner import BaseRunner
from src.utils.config import Settings


class ClaudeRunner(BaseRunner):
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, model_id: str, settings: Settings) -> None:
        super().__init__(model_id)
        self._retry_cfg = settings.retry
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._instructor_client: instructor.Instructor = instructor.from_anthropic(
            client=self._client,
            mode=instructor.Mode.ANTHROPIC_TOOLS,
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
            return self._client.messages.create(
                model=self.model_id,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

        start = time.monotonic()
        response = _call()
        elapsed_ms = (time.monotonic() - start) * 1000

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = response.usage
        return RunResult(
            model_id=self.model_id,
            prompt_id=prompt_id,
            output_text=text,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
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
                max_tokens=self.DEFAULT_MAX_TOKENS,
            )

        start = time.monotonic()
        parsed, raw_response = _call()
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = raw_response.usage
        return RunResult(
            model_id=self.model_id,
            prompt_id=prompt_id,
            output_text=parsed.model_dump_json(),
            output_parsed=parsed,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            latency_ms=elapsed_ms,
        )
