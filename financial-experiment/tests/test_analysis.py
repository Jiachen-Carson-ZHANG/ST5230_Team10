"""Unit tests for analysis metric functions using synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    enforce_mandatory_field_quality_gate,
    compute_flip_rate,
    compute_entropy,
    compute_conservatism_score,
    compute_action_urgency_score,
    compute_unsupported_claim_rate,
)

# Monkey-patch FIGURES_DIR to a temp location so tests don't write to real data/
import src.analysis as analysis_module
analysis_module.FIGURES_DIR = Path("/tmp/test_figures")


def _make_df(rows):
    """Build a DataFrame from a list of dicts with sensible defaults."""
    defaults = {
        "article_id": "01",
        "persona_id": "baseline",
        "model": "openai/gpt-4o",
        "sample_idx": 0,
        "raw_response": "test response",
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


def test_mandatory_field_quality_gate_passes_when_threshold_met():
    df = _make_df([
        {"risk_rating_score": 3, "strategic_action": "Hold_Monitor"},
        {"risk_rating_score": 4, "strategic_action": "Reduce_Exposure"},
        {"risk_rating_score": 2, "strategic_action": "Strong_Buy"},
        {"risk_rating_score": None, "strategic_action": None},
    ])

    rate = enforce_mandatory_field_quality_gate(df, threshold=0.75)

    assert rate == 0.75


def test_mandatory_field_quality_gate_raises_below_threshold():
    df = _make_df([
        {"risk_rating_score": 3, "strategic_action": "Hold_Monitor"},
        {"risk_rating_score": None, "strategic_action": None},
        {"risk_rating_score": None, "strategic_action": None},
        {"risk_rating_score": None, "strategic_action": None},
    ])

    with pytest.raises(SystemExit):
        enforce_mandatory_field_quality_gate(df, threshold=0.75)


# ── compute_flip_rate ────────────────────────────────────────────────────

def test_flip_rate_all_same():
    """All 5 posture labels agree → flip rate = 0."""
    rows = [{"strategic_action": "Hold_Monitor", "sample_idx": i} for i in range(5)]
    df = _make_df(rows)
    result = compute_flip_rate(df)
    assert result["flip_rate"].iloc[0] == 0.0

def test_flip_rate_uses_strategic_posture_not_raw_action():
    """Actions that collapse to the same posture should not count as flips."""
    actions = ["Strong_Buy", "Clear_Short", "Strong_Buy", "Clear_Short"]
    rows = [{"strategic_action": a, "sample_idx": i} for i, a in enumerate(actions)]
    df = _make_df(rows)
    result = compute_flip_rate(df)
    assert result["flip_rate"].iloc[0] == 0.0

def test_flip_rate_majority():
    """3 of 4 posture labels agree → flip rate = 0.25."""
    rows = [
        {"strategic_action": "Hold_Monitor", "sample_idx": 0},
        {"strategic_action": "Hold_Monitor", "sample_idx": 1},
        {"strategic_action": "Hold_Monitor", "sample_idx": 2},
        {"strategic_action": "Reduce_Exposure", "sample_idx": 3},
    ]
    df = _make_df(rows)
    result = compute_flip_rate(df)
    assert result["flip_rate"].iloc[0] == 0.25


# ── compute_entropy ──────────────────────────────────────────────────────

def test_entropy_all_same():
    """All samples pick the same action → entropy = 0."""
    rows = [{"strategic_action": "Hold_Monitor", "sample_idx": i} for i in range(5)]
    df = _make_df(rows)
    result = compute_entropy(df)
    assert result["entropy"].iloc[0] == 0.0

def test_entropy_uniform():
    """Equal distribution across all 5 actions → max entropy = log2(5)."""
    actions = ["Strong_Buy", "Hold_Monitor", "Reduce_Exposure", "Clear_Short", "Halt_Compliance"]
    rows = [{"strategic_action": a, "sample_idx": i} for i, a in enumerate(actions)]
    df = _make_df(rows)
    result = compute_entropy(df)
    expected = np.log2(5)
    assert abs(result["entropy"].iloc[0] - expected) < 0.01


# ── compute_conservatism_score ───────────────────────────────────────────

def test_conservatism_score_mean():
    """Mean of [1, 3, 5] = 3.0."""
    rows = [
        {"risk_rating_score": 1, "sample_idx": 0},
        {"risk_rating_score": 3, "sample_idx": 1},
        {"risk_rating_score": 5, "sample_idx": 2},
    ]
    df = _make_df(rows)
    result = compute_conservatism_score(df)
    assert result.iloc[0, 0] == 3.0


# ── compute_action_urgency_score ─────────────────────────────────────────

def test_action_urgency_score_mean_matches_friend_examples():
    """Immediate should score higher than Short_Term urgency."""
    rows = [
        {"action_urgency": "Immediate", "sample_idx": 0},
        {"action_urgency": "Short_Term", "sample_idx": 1},
        {"action_urgency": "Long_Term", "sample_idx": 2},
    ]
    df = _make_df(rows)
    result = compute_action_urgency_score(df)
    assert result.iloc[0, 0] == pytest.approx((5 + 2 + 1) / 3)

def test_action_urgency_score_short_term_maps_to_two():
    """The friend's example treats monitored near-term action as low urgency, not mid-scale."""
    rows = [{"action_urgency": "Short_Term", "sample_idx": i} for i in range(3)]
    df = _make_df(rows)
    result = compute_action_urgency_score(df)
    assert result.iloc[0, 0] == 2.0


# ── compute_unsupported_claim_rate ───────────────────────────────────────

def test_unsupported_claim_rate_mixed():
    """2 of 4 flagged → 50%."""
    rows = [
        {"unsupported_financial_claim_flag": True, "sample_idx": 0},
        {"unsupported_financial_claim_flag": False, "sample_idx": 1},
        {"unsupported_financial_claim_flag": True, "sample_idx": 2},
        {"unsupported_financial_claim_flag": False, "sample_idx": 3},
    ]
    df = _make_df(rows)
    result = compute_unsupported_claim_rate(df)
    assert result.iloc[0, 0] == 0.5
    exported = analysis_module.FIGURES_DIR / "unsupported_claim_cases.csv"
    assert exported.exists()
    exported_lines = exported.read_text().strip().splitlines()
    assert len(exported_lines) == 3  # header + 2 flagged cases

def test_unsupported_claim_rate_string_booleans():
    """Analysis must handle string 'True'/'False' from CSV."""
    rows = [
        {"unsupported_financial_claim_flag": "True", "sample_idx": 0},
        {"unsupported_financial_claim_flag": "False", "sample_idx": 1},
    ]
    df = _make_df(rows)
    result = compute_unsupported_claim_rate(df)
    assert result.iloc[0, 0] == 0.5
