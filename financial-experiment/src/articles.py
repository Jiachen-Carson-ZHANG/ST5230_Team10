"""Helpers for normalizing and loading experiment articles."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_article_record(record):
    """Normalize either source-corpus or pipeline-shaped article records."""
    article_id = record.get("article_id", record.get("id"))
    article_text = record.get("article_text", record.get("content"))

    if article_id is None or article_text is None:
        raise ValueError(
            "Each article record must include either article_id/article_text "
            "or id/content."
        )

    normalized = dict(record)
    normalized["article_id"] = article_id
    normalized["article_text"] = article_text
    normalized.setdefault("headline", "")
    normalized.setdefault("ticker", "")
    normalized.setdefault("source", "")
    normalized.setdefault("date", "")
    normalized.setdefault("metadata", {})
    return normalized


def load_articles_file(path):
    """Load a JSON article list and normalize every record to one shared shape."""
    path = Path(path)
    with open(path) as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Articles file must contain a JSON list.")

    return [normalize_article_record(record) for record in records]
