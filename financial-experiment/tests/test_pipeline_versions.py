"""Tests for pipeline version guards and smoke-test record shape."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.test_run as test_run
from src.extractor import EXTRACTOR_EXPERIMENT_VERSION, validate_results_file_version
from src.generator import GENERATOR_EXPERIMENT_VERSION, validate_response_file_version


def test_validate_response_file_version_rejects_mismatch(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps(
            {
                "response_id": "01__baseline__openai-gpt-4o__00",
                "generator_version": "old-generator-version",
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="generator_version"):
        validate_response_file_version(path)


def test_validate_results_file_version_rejects_mismatch(tmp_path):
    path = tmp_path / "results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["response_id", "extractor_version", "risk_rating_score"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "response_id": "01__baseline__openai-gpt-4o__00",
                "extractor_version": "old-extractor-version",
                "risk_rating_score": 3,
            }
        )

    with pytest.raises(RuntimeError, match="extractor_version"):
        validate_results_file_version(path)


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
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def test_run_generator_test_writes_versioned_response_records(tmp_path, monkeypatch):
    monkeypatch.setattr(test_run, "TEST_DIR", tmp_path)
    monkeypatch.setattr(test_run, "TEST_RESPONSES", tmp_path / "test_responses.jsonl")
    monkeypatch.setattr(test_run, "TEST_PERSONAS", ["baseline"])

    article = {
        "article_id": "01",
        "ticker": "GME.N",
        "headline": "GameStop headline",
        "article_text": "GameStop article body.",
    }
    fake_client = _FakeClient(
        "[Risk Rating]: Moderate Risk\n"
        "[Strategic Action]: Hold and Monitor\n"
        "[Justification]: The situation remains uncertain."
    )

    responses = test_run.run_generator_test(fake_client, [article])

    assert responses[0]["generator_version"] == GENERATOR_EXPERIMENT_VERSION
    assert responses[0]["ticker"] == article["ticker"]
    assert responses[0]["headline"] == article["headline"]
    assert responses[0]["response_id"] == "01__baseline__openai-gpt-4o__00"
