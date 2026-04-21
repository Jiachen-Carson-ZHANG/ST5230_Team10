"""Normalize the source corpus into data/news/articles.json for the pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.articles import load_articles_file
from src.config_runtime import ARTICLES_PATH


DEFAULT_SOURCE_PATH = Path(__file__).resolve().parents[2] / "news_corpus_final.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize the source news corpus into the pipeline article schema."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Path to the source corpus JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTICLES_PATH,
        help="Where to write the normalized articles.json file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: Source corpus not found at {args.input}")
        sys.exit(1)

    articles = load_articles_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(articles)} normalized articles to {args.output}")


if __name__ == "__main__":
    main()
