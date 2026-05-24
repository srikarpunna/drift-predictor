from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Type

from pydantic import BaseModel as PydanticBaseModel

from src.models import RunResult


class BaseRunner(ABC):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def run(
        self,
        prompt: str,
        prompt_id: str,
        schema: Optional[Type[PydanticBaseModel]] = None,
    ) -> RunResult:
        if schema is not None:
            return self.run_structured(prompt, prompt_id, schema)
        return self.run_text(prompt, prompt_id)

    def run_text(self, prompt: str, prompt_id: str) -> RunResult:
        start = time.monotonic()
        try:
            return self._run_text_impl(prompt, prompt_id)
        except Exception as exc:
            return RunResult(
                model_id=self.model_id,
                prompt_id=prompt_id,
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=(time.monotonic() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_structured(
        self,
        prompt: str,
        prompt_id: str,
        schema: Type[PydanticBaseModel],
    ) -> RunResult:
        start = time.monotonic()
        try:
            return self._run_structured_impl(prompt, prompt_id, schema)
        except Exception as exc:
            return RunResult(
                model_id=self.model_id,
                prompt_id=prompt_id,
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=(time.monotonic() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

    @abstractmethod
    def _run_text_impl(self, prompt: str, prompt_id: str) -> RunResult:
        ...

    @abstractmethod
    def _run_structured_impl(
        self,
        prompt: str,
        prompt_id: str,
        schema: Type[PydanticBaseModel],
    ) -> RunResult:
        ...
