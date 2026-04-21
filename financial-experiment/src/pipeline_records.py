"""Shared helpers for pipeline record shapes and resumability guards."""

import csv
import json


GENERATOR_EXPERIMENT_VERSION = "2026-03-27-freeform-generator-v1"
EXTRACTOR_EXPERIMENT_VERSION = "2026-03-28-freeform-extractor-v4"


def make_response_id(article_id, persona_id, model_id, sample_idx):
    """Build a stable response ID for one article/persona/model/sample tuple."""
    model_slug = model_id.replace("/", "-")
    return f"{article_id}__{persona_id}__{model_slug}__{sample_idx:02d}"


def build_response_record(
    article,
    persona_id,
    model_id,
    sample_idx,
    raw_response,
    *,
    prompt_tokens=None,
    completion_tokens=None,
    estimated_cost_usd=None,
):
    """Build the shared response record written by generator and smoke test."""
    return {
        "response_id": make_response_id(
            article["article_id"], persona_id, model_id, sample_idx
        ),
        "generator_version": GENERATOR_EXPERIMENT_VERSION,
        "article_id": article["article_id"],
        "persona_id": persona_id,
        "model": model_id,
        "sample_idx": sample_idx,
        "raw_response": raw_response,
        "article_text": article["article_text"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "ticker": article.get("ticker", ""),
        "headline": article.get("headline", ""),
    }


def validate_response_file_version(path):
    """Abort resumability if responses.jsonl was produced by a different contract."""
    if not path.exists():
        return

    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            version = record.get("generator_version")
            if version != GENERATOR_EXPERIMENT_VERSION:
                raise RuntimeError(
                    f"{path} contains generator_version={version!r} on line "
                    f"{line_number}, expected {GENERATOR_EXPERIMENT_VERSION!r}. "
                    "Archive or delete the old responses file before resuming."
                )


def validate_results_file_version(path):
    """Abort resumability if results.csv was produced by a different rubric version."""
    if not path.exists():
        return

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        if "extractor_version" not in reader.fieldnames:
            raise RuntimeError(
                f"{path} is missing the extractor_version column. "
                "Archive or delete the old results file before resuming."
            )

        for row_number, row in enumerate(reader, start=2):
            version = row.get("extractor_version")
            if version != EXTRACTOR_EXPERIMENT_VERSION:
                raise RuntimeError(
                    f"{path} contains extractor_version={version!r} on line "
                    f"{row_number}, expected {EXTRACTOR_EXPERIMENT_VERSION!r}. "
                    "Archive or delete the old results file before resuming."
                )
