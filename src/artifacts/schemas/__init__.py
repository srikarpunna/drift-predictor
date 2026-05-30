"""
Schema registry — auto-discovers Pydantic schemas from this folder.

Convention: one root schema class per file. File stem = lookup key.
  interview_evaluation.py  →  get_schema("interview_evaluation")
  support_audit.py         →  get_schema("support_audit")
  diagnostic.py            →  get_schema("diagnostic")

Add a new schema file → automatically available. No manual registration.
"""

import importlib.util
import inspect
from pathlib import Path

from pydantic import BaseModel

_SCHEMAS_DIR = Path(__file__).parent


def _build_registry() -> dict[str, type[BaseModel]]:
    registry: dict[str, type[BaseModel]] = {}
    for py_file in sorted(_SCHEMAS_DIR.glob("*.py")):
        if py_file.stem == "__init__":
            continue
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # Collect top-level BaseModel subclasses (no dot in __qualname__ = not nested)
        top_level = [
            obj
            for _, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, BaseModel)
            and obj is not BaseModel
            and obj.__module__ == mod.__name__
            and "." not in obj.__qualname__
        ]

        if len(top_level) == 1:
            # Convention: single root class → register by file stem
            registry[py_file.stem] = top_level[0]
        else:
            # Multiple top-level classes → register each by class name
            for cls in top_level:
                registry[cls.__name__] = cls

    return registry


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = _build_registry()


def get_schema(name: str) -> type[BaseModel]:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"Schema '{name}' not found. Available: {sorted(SCHEMA_REGISTRY)}"
        )
    return SCHEMA_REGISTRY[name]
