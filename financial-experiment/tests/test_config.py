"""Tests for import-safe config handling."""

import importlib.util
from pathlib import Path

import pytest


CONFIG_PATH = Path(__file__).parent.parent / "src" / "config_runtime.py"


def load_config_module():
    spec = importlib.util.spec_from_file_location("isolated_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_config_import_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ST5230_FORCE_CONFIG_EXAMPLE", "1")

    config = load_config_module()

    assert config.OPENROUTER_API_KEY is None


def test_require_openrouter_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ST5230_FORCE_CONFIG_EXAMPLE", "1")

    config = load_config_module()

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        config.require_openrouter_api_key()
