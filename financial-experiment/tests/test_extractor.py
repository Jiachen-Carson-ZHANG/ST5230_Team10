"""Unit tests for extractor Pydantic validators and helpers."""

import sys
from pathlib import Path

# Allow imports from src/ without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import validators and schemas directly — no API key needed
from src.extractor import (
    _coerce_bool,
    _coerce_int_1_5,
    ExtractionResult,
    HallucinationResult,
    HALLUCINATION_PROMPT,
    VALID_STRATEGIC_ACTIONS,
    VALID_URGENCIES,
    VALID_FOCUSES,
    VALID_BASES,
)


# ── _coerce_int_1_5 ──────────────────────────────────────────────────────

def test_coerce_int_none():
    assert _coerce_int_1_5(None) is None

def test_coerce_int_normal():
    assert _coerce_int_1_5(3) == 3

def test_coerce_int_string():
    assert _coerce_int_1_5("4") == 4

def test_coerce_int_clamp_high():
    assert _coerce_int_1_5(7) == 5

def test_coerce_int_clamp_low():
    assert _coerce_int_1_5(0) == 1

def test_coerce_int_invalid_string():
    assert _coerce_int_1_5("abc") is None

def test_coerce_int_float():
    assert _coerce_int_1_5(3.7) == 3  # int() truncates


# ── _coerce_bool ──────────────────────────────────────────────────────────

def test_coerce_bool_none():
    assert _coerce_bool(None) is None

def test_coerce_bool_true():
    assert _coerce_bool(True) is True

def test_coerce_bool_false():
    assert _coerce_bool(False) is False

def test_coerce_bool_string_true():
    assert _coerce_bool("true") is True

def test_coerce_bool_string_false():
    assert _coerce_bool("false") is False

def test_coerce_bool_string_yes():
    assert _coerce_bool("yes") is True

def test_coerce_bool_string_one():
    assert _coerce_bool("1") is True

def test_coerce_bool_string_zero():
    assert _coerce_bool("0") is False


# ── ExtractionResult — full model validation ──────────────────────────────

def test_extraction_result_happy_path():
    data = {
        "risk_rating_score": 3,
        "strategic_action": "Hold_Monitor",
        "action_urgency": "Short_Term",
        "compliance_refusal_flag": False,
        "analysis_primary_focus": "Fundamentals",
        "reasoning_basis": "News_Fact_Driven",
        "tone_confidence_level": 3,
        "risk_thesis_hook": "Revenue stable pending next quarter",
    }
    result = ExtractionResult.model_validate(data)
    assert result.risk_rating_score == 3
    assert result.strategic_action == "Hold_Monitor"
    assert result.compliance_refusal_flag is False

def test_extraction_result_string_coercion():
    """LLM sometimes returns integers as strings — validators should handle this."""
    data = {
        "risk_rating_score": "4",
        "tone_confidence_level": "2",
        "compliance_refusal_flag": "true",
    }
    result = ExtractionResult.model_validate(data)
    assert result.risk_rating_score == 4
    assert result.tone_confidence_level == 2
    assert result.compliance_refusal_flag is True

def test_extraction_result_invalid_enum_becomes_none():
    """Invalid enum values should become None, not crash the whole validation."""
    data = {
        "risk_rating_score": 3,
        "strategic_action": "INVALID_ACTION",
        "action_urgency": "INVALID_URGENCY",
        "analysis_primary_focus": "INVALID_FOCUS",
        "reasoning_basis": "INVALID_BASIS",
    }
    result = ExtractionResult.model_validate(data)
    assert result.risk_rating_score == 3  # valid field kept
    assert result.strategic_action is None
    assert result.action_urgency is None
    assert result.analysis_primary_focus is None
    assert result.reasoning_basis is None

def test_extraction_result_empty_dict():
    """Empty dict should produce all-None fields, not crash."""
    result = ExtractionResult.model_validate({})
    assert result.risk_rating_score is None
    assert result.strategic_action is None

def test_extraction_result_all_valid_enums():
    """Every valid enum value should pass validation."""
    for action in VALID_STRATEGIC_ACTIONS:
        r = ExtractionResult.model_validate({"strategic_action": action})
        assert r.strategic_action == action

    for urgency in VALID_URGENCIES:
        r = ExtractionResult.model_validate({"action_urgency": urgency})
        assert r.action_urgency == urgency

    for focus in VALID_FOCUSES:
        r = ExtractionResult.model_validate({"analysis_primary_focus": focus})
        assert r.analysis_primary_focus == focus

    for basis in VALID_BASES:
        r = ExtractionResult.model_validate({"reasoning_basis": basis})
        assert r.reasoning_basis == basis


# ── HallucinationResult ──────────────────────────────────────────────────

def test_hallucination_result_true():
    result = HallucinationResult.model_validate(
        {
            "step_by_step_verification": "Only predictions/opinions found",
            "unsupported_financial_claim_flag": True,
        }
    )
    assert result.unsupported_financial_claim_flag is True
    assert result.step_by_step_verification == "Only predictions/opinions found"

def test_hallucination_result_string_false():
    result = HallucinationResult.model_validate(
        {
            "step_by_step_verification": "Only predictions/opinions found",
            "unsupported_financial_claim_flag": "false",
        }
    )
    assert result.unsupported_financial_claim_flag is False

def test_hallucination_result_empty():
    result = HallucinationResult.model_validate({})
    assert result.unsupported_financial_claim_flag is None


def test_hallucination_prompt_allows_inference_and_defaults_borderline_to_false():
    assert "Reasoning process" in HALLUCINATION_PROMPT
    assert "Only predictions/opinions found" in HALLUCINATION_PROMPT
    assert "reasonable inference" in HALLUCINATION_PROMPT
    assert "return false" in HALLUCINATION_PROMPT
    assert "If you are unsure" in HALLUCINATION_PROMPT
    assert "market impact" in HALLUCINATION_PROMPT
    assert "EXACTLY two keys" in HALLUCINATION_PROMPT
