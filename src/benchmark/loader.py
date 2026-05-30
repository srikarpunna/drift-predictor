"""
Loads benchmark configuration and prompt sets from config.json.

Usage:
    cfg = load_config()
    prompts = load_prompt_set(cfg.prompt_sets[0])

Prompt discovery: scans folder for *.txt files.
Filename convention: {schema_name}-{NNN}.txt
  interview_evaluation-001.txt  →  schema "interview_evaluation", id "interview_evaluation-001"
  support_audit-003.txt         →  schema "support_audit",        id "support_audit-003"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.benchmark.prompt_item import PromptItem

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"

_FILENAME_RE = re.compile(r"^(.+)-(\d+)$")


@dataclass
class ModelPair:
    name: str
    provider: str
    old_model: str
    new_model: str


@dataclass
class PromptSetConfig:
    path: Path


@dataclass
class BenchmarkConfig:
    schema_path: Path
    prompt_sets: list[PromptSetConfig]
    model_pairs: list[ModelPair]
    output_dir: Path
    taxonomy_labels: list[str]
    verbose: bool
    save_outputs: bool

    def get_pair(self, name: str) -> ModelPair:
        for pair in self.model_pairs:
            if pair.name == name:
                return pair
        raise KeyError(f"Model pair '{name}' not in config. Available: {[p.name for p in self.model_pairs]}")


def load_config(config_path: Path = _CONFIG_PATH) -> BenchmarkConfig:
    raw = json.loads(config_path.read_text())
    return BenchmarkConfig(
        schema_path=_PROJECT_ROOT / raw["schema_path"],
        prompt_sets=[
            PromptSetConfig(path=_PROJECT_ROOT / ps["path"])
            for ps in raw["prompt_sets"]
        ],
        model_pairs=[
            ModelPair(
                name=p["name"],
                provider=p["provider"],
                old_model=p["old_model"],
                new_model=p["new_model"],
            )
            for p in raw["model_pairs"]
        ],
        output_dir=_PROJECT_ROOT / raw["output_dir"],
        taxonomy_labels=raw["taxonomy_labels"],
        verbose=raw.get("run", {}).get("verbose", True),
        save_outputs=raw.get("run", {}).get("save_outputs", True),
    )


def load_prompt_set(prompt_set: PromptSetConfig) -> list[PromptItem]:
    """
    Scans folder for *.txt files. Derives schema name from filename.
    Filename must match pattern: {schema_name}-{NNN}.txt
    """
    folder = prompt_set.path
    prompts: list[PromptItem] = []

    for txt_file in sorted(folder.glob("*.txt")):
        match = _FILENAME_RE.match(txt_file.stem)
        if not match:
            continue  # skip files that don't match convention
        schema_name = match.group(1)
        prompt_text = txt_file.read_text().strip()

        prompts.append(PromptItem(
            id=txt_file.stem,
            output_schema=schema_name,
            prompt_text=prompt_text,
        ))

    return prompts


def load_all_prompts(cfg: BenchmarkConfig) -> list[PromptItem]:
    """Load all prompt sets combined."""
    all_prompts: list[PromptItem] = []
    for ps in cfg.prompt_sets:
        all_prompts.extend(load_prompt_set(ps))
    return all_prompts
