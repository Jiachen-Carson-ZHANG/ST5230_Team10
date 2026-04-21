"""Tests for the final generator prompt contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import (
    MAX_RESPONSE_WORDS,
    PERSONAS,
    REQUIRED_SECTION_HEADINGS,
    build_user_prompt,
)


SAMPLE_ARTICLE = {
    "article_id": "05",
    "ticker": "GME.N",
    "headline": "GameStop posts 14% fall in quarterly revenue amid digital gaming shift",
    "article_text": "GameStop reported a 14% drop in revenue while maintaining profitability.",
}


def test_baseline_persona_is_empty_control():
    assert PERSONAS["baseline"] == ""


def test_new_persona_set_is_present():
    assert set(PERSONAS) == {
        "baseline",
        "conservative_officer",
        "aggressive_hedge_fund",
        "neutral_researcher",
    }


def test_user_prompt_includes_ticker_and_article_context():
    prompt = build_user_prompt(SAMPLE_ARTICLE)
    assert "SPECIFICALLY for GME.N" in prompt
    assert SAMPLE_ARTICLE["headline"] in prompt
    assert SAMPLE_ARTICLE["article_text"] in prompt


def test_user_prompt_requires_exact_section_headings():
    prompt = build_user_prompt(SAMPLE_ARTICLE)
    for heading in REQUIRED_SECTION_HEADINGS:
        assert heading in prompt


def test_user_prompt_sets_free_form_not_json_contract():
    prompt = build_user_prompt(SAMPLE_ARTICLE)
    assert "json" not in prompt.lower()
    assert f"under {MAX_RESPONSE_WORDS} words" in prompt.lower()
    assert "focus entirely on our holding" in prompt.lower()
