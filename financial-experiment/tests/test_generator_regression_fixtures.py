"""Offline regression fixtures for representative generator outputs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import ExtractionResult, HallucinationResult
from src.prompts import inspect_response_format


FIXTURES_PATH = Path(__file__).parent / "fixtures" / "generator_regression_cases.json"


def test_generator_regression_fixtures_cover_representative_outputs():
    cases = json.loads(FIXTURES_PATH.read_text())

    assert len(cases) >= 4

    for case in cases:
        format_result = inspect_response_format(case["raw_response"])
        assert format_result["has_required_headings"] is case["expected"]["has_required_headings"]
        assert format_result["word_limit_status"] == case["expected"]["word_limit_status"]

        extracted = ExtractionResult.model_validate(case["expected_extraction"])
        hallucination = HallucinationResult.model_validate(case["expected_hallucination"])

        assert extracted.strategic_action is not None
        assert extracted.risk_rating_score is not None
        assert hallucination.unsupported_financial_claim_flag is not None
