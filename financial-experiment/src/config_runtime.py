"""Runtime config loader with a safe fallback for public repo use."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


USING_FALLBACK_CONFIG = False


def _load_config_example():
    example_path = Path(__file__).with_name("config.example.py")
    spec = importlib.util.spec_from_file_location("src.config_example_runtime", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fallback config from {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    if os.environ.get("ST5230_FORCE_CONFIG_EXAMPLE") == "1":
        raise ModuleNotFoundError
    from src.config import *  # type: ignore  # noqa: F401,F403
except ModuleNotFoundError:
    fallback = _load_config_example()
    USING_FALLBACK_CONFIG = True
    for name in dir(fallback):
        if name.startswith("_"):
            continue
        globals()[name] = getattr(fallback, name)
