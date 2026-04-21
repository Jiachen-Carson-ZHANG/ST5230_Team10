"""Tests for generator free-form output format checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import inspect_response_format


VALID_RESPONSE = """[Risk Rating]: Moderate Risk
[Strategic Action]: Hold and Monitor
[Justification]: Revenue pressure is real, but the company remains profitable and the article does not show a near-term collapse."""


def _build_response_with_word_count(word_count):
    body = " ".join(f"word{i}" for i in range(word_count))
    return (
        "[Risk Rating]: Moderate Risk\n"
        "[Strategic Action]: Hold and Monitor\n"
        f"[Justification]: {body}"
    )


def test_inspect_response_format_accepts_required_headings():
    result = inspect_response_format(VALID_RESPONSE)

    assert result["has_required_headings"] is True
    assert result["missing_headings"] == []


def test_inspect_response_format_rejects_missing_heading():
    result = inspect_response_format("[Risk Rating]: High Risk\n[Justification]: Missing the action section.")

    assert result["has_required_headings"] is False
    assert "[Strategic Action]:" in result["missing_headings"]


def test_inspect_response_format_counts_words_and_limit():
    result = inspect_response_format(VALID_RESPONSE)

    assert result["word_count"] > 0
    assert result["within_word_limit"] is True


def test_inspect_response_format_warns_on_small_overflow():
    result = inspect_response_format(_build_response_with_word_count(205))

    assert result["word_limit_status"] == "warn"
    assert result["within_word_limit"] is True


def test_inspect_response_format_fails_on_large_overflow():
    result = inspect_response_format(_build_response_with_word_count(245))

    assert result["word_limit_status"] == "fail"
    assert result["within_word_limit"] is False
