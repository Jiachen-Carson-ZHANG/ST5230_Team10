"""Tests for extractor failure diagnostics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import (
    EXTRACTION_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    ExtractionResult,
    call_extractor,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, mode):
        self._mode = mode

    def create(self, **kwargs):
        if self._mode == "raise":
            raise RuntimeError("network issue")
        if self._mode == "invalid_json":
            return _FakeResponse("not-json")
        if self._mode == "schema_error":
            return _FakeResponse(json.dumps({"risk_thesis_hook": ["not", "a", "string"]}))
        raise AssertionError(f"Unknown mode: {self._mode}")


class _FakeChat:
    def __init__(self, mode):
        self.completions = _FakeCompletions(mode)


class _FakeClient:
    def __init__(self, mode):
        self.chat = _FakeChat(mode)


def test_call_extractor_reports_api_error_diagnostics():
    result, diagnostics = call_extractor(
        _FakeClient("raise"),
        EXTRACTOR_SYSTEM_PROMPT,
        EXTRACTION_PROMPT.format(response_text="sample"),
        ExtractionResult,
        include_diagnostics=True,
    )

    assert result == {}
    assert diagnostics["error_type"] == "api_error"


def test_call_extractor_reports_invalid_json_diagnostics():
    result, diagnostics = call_extractor(
        _FakeClient("invalid_json"),
        EXTRACTOR_SYSTEM_PROMPT,
        EXTRACTION_PROMPT.format(response_text="sample"),
        ExtractionResult,
        include_diagnostics=True,
    )

    assert result == {}
    assert diagnostics["error_type"] == "invalid_json"


def test_call_extractor_reports_schema_validation_diagnostics():
    result, diagnostics = call_extractor(
        _FakeClient("schema_error"),
        EXTRACTOR_SYSTEM_PROMPT,
        EXTRACTION_PROMPT.format(response_text="sample"),
        ExtractionResult,
        include_diagnostics=True,
    )

    assert result == {}
    assert diagnostics["error_type"] == "schema_validation_error"
