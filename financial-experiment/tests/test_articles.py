"""Tests for article normalization and shared corpus loading."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.articles import load_articles_file, normalize_article_record


def test_normalize_article_record_accepts_pipeline_shape():
    record = {
        "article_id": "01",
        "article_text": "Article text",
        "ticker": "GOOGL.O",
        "headline": "Headline",
        "source": "Reuters",
        "date": "2026-03-23",
        "metadata": {"expected_ambiguity": "medium"},
    }

    normalized = normalize_article_record(record)

    assert normalized["article_id"] == "01"
    assert normalized["article_text"] == "Article text"
    assert normalized["ticker"] == "GOOGL.O"


def test_normalize_article_record_accepts_source_corpus_shape():
    record = {
        "id": "09",
        "content": "Source corpus text",
        "ticker": "JBS",
        "headline": "JBS posts flat Q4 profit on record sales but lower US beef margins",
        "source": "Reuters",
        "date": "2026-03-25",
        "metadata": {"expected_ambiguity": "high"},
    }

    normalized = normalize_article_record(record)

    assert normalized["article_id"] == "09"
    assert normalized["article_text"] == "Source corpus text"
    assert normalized["headline"].startswith("JBS posts")


def test_normalize_article_record_rejects_unknown_shape():
    with pytest.raises(ValueError, match="article_id|id"):
        normalize_article_record({"headline": "Missing identifiers"})


def test_load_articles_file_normalizes_all_records(tmp_path):
    path = tmp_path / "articles.json"
    path.write_text(json.dumps([
        {"id": "01", "content": "Alpha", "ticker": "A", "headline": "H1"},
        {"article_id": "02", "article_text": "Beta", "ticker": "B", "headline": "H2"},
    ]))

    records = load_articles_file(path)

    assert [record["article_id"] for record in records] == ["01", "02"]
    assert [record["article_text"] for record in records] == ["Alpha", "Beta"]
