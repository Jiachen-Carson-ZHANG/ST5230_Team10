"""
Smoke test — run a few real API calls before launching the full experiment.

Tests ONE article × 4 personas × GPT-4o × 1 sample. Then immediately runs the
extractor on those responses so you can verify the full pipeline end-to-end.

Total cost: low single-digit cents. Saves outputs to data/test/ (separate from
main data dirs).

Run: python3 src/test_run.py
"""

import json
import sys
import csv
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_runtime import (
    OPENROUTER_BASE_URL,
    MODEL_PARAMS, GPT4O_ID,
    require_openrouter_api_key,
)
from src.prompts import (
    PERSONAS,
    build_system_prompt,
    build_user_prompt,
    inspect_response_format,
)
from src.generator import validate_prompts, load_articles
from src.pipeline_records import build_response_record
from src.extractor import (
    EXTRACTOR_SYSTEM_PROMPT, EXTRACTION_PROMPT, HALLUCINATION_PROMPT,
    build_results_row,
    call_extractor,
    CSV_FIELDS,
    ExtractionResult, HallucinationResult,
)

# ── Test config ────────────────────────────────────────────────────────────
TEST_DIR       = Path(__file__).parent.parent / "data" / "test"
TEST_RESPONSES = TEST_DIR / "test_responses.jsonl"
TEST_RESULTS   = TEST_DIR / "test_results.csv"

# Which article to test on (index into articles list)
TEST_ARTICLE_IDX = 0

# Test the full condition set so the smoke test matches the full run contract.
TEST_PERSONAS = [
    "baseline",
    "conservative_officer",
    "aggressive_hedge_fund",
    "neutral_researcher",
]

# Only test with GPT-4o (fastest feedback, most reliable output format)
TEST_MODEL = GPT4O_ID


def run_generator_test(client, articles):
    article = articles[TEST_ARTICLE_IDX]
    headline = article.get("headline", "")

    print(f"\n{'='*60}")
    print(f"TEST ARTICLE: {article['article_id']}")
    print(f"Headline: {headline}")
    print(f"{'='*60}\n")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    responses = []

    with open(TEST_RESPONSES, "w") as f:
        for persona_id in TEST_PERSONAS:
            system_prompt = build_system_prompt(persona_id)
            user_prompt = build_user_prompt(article)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            print(f"--- Calling GPT-4o | persona: {persona_id} ---")
            try:
                resp = client.chat.completions.create(
                    model=TEST_MODEL,
                    messages=messages,
                    **MODEL_PARAMS[TEST_MODEL],
                )
                raw_response = resp.choices[0].message.content
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            usage = getattr(resp, "usage", None)
            record = build_response_record(
                article,
                persona_id,
                TEST_MODEL,
                0,
                raw_response,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                estimated_cost_usd=None,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            responses.append(record)

            format_check = inspect_response_format(raw_response)
            # Print the full response so you can visually inspect it
            print(f"\n[RESPONSE — {persona_id}]")
            print(raw_response)
            print(
                "FORMAT CHECK:",
                f"headings_ok={format_check['has_required_headings']}",
                f"word_limit_status={format_check['word_limit_status']}",
                f"within_word_limit={format_check['within_word_limit']}",
                f"within_target_word_limit={format_check['within_target_word_limit']}",
                f"word_count={format_check['word_count']}",
            )
            if format_check["missing_headings"]:
                print(f"  missing_headings: {format_check['missing_headings']}")
            print()

    print(f"\nGenerator test done. {len(responses)} responses saved to {TEST_RESPONSES}")
    return responses


def run_extractor_test(client, responses):
    print(f"\n{'='*60}")
    print("RUNNING EXTRACTOR ON TEST RESPONSES")
    print(f"{'='*60}\n")

    rows = []
    extraction_successes = 0
    for resp in responses:
        raw_response = resp["raw_response"]

        extracted, extraction_diagnostics = call_extractor(
            client,
            EXTRACTOR_SYSTEM_PROMPT,
            EXTRACTION_PROMPT.format(response_text=raw_response),
            ExtractionResult,
            include_diagnostics=True,
        )
        hallucination, hallucination_diagnostics = call_extractor(
            client,
            EXTRACTOR_SYSTEM_PROMPT,
            HALLUCINATION_PROMPT.format(
                article_text=resp["article_text"],
                response_text=raw_response,
            ),
            HallucinationResult,
            include_diagnostics=True,
        )

        row = build_results_row(
            resp,
            extracted,
            hallucination,
            extraction_diagnostics=extraction_diagnostics,
            hallucination_diagnostics=hallucination_diagnostics,
        )
        rows.append(row)
        mandatory_fields_ok = (
            row["risk_rating_score"] is not None
            and row["strategic_action"] is not None
        )
        extraction_successes += int(mandatory_fields_ok)

        # Print a concise summary so you can see what was extracted
        print(f"[EXTRACTED — {resp['persona_id']}]")
        for key, val in row.items():
            if key not in ("response_id", "article_id", "model", "sample_idx"):
                print(f"  {key}: {val}")
        print(f"  mandatory_fields_ok: {mandatory_fields_ok}")
        print()

    # Save to CSV
    with open(TEST_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extractor test done. Results saved to {TEST_RESULTS}")
    print(
        "EXTRACTION SUMMARY:",
        f"{extraction_successes}/{len(responses)} responses returned both "
        "risk_rating_score and strategic_action.",
    )


def main():
    validate_prompts()
    articles = load_articles()

    client = OpenAI(
        api_key=require_openrouter_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

    responses = run_generator_test(client, articles)
    if not responses:
        print("No responses generated — check API key and model access.")
        sys.exit(1)

    run_extractor_test(client, responses)

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print("Review the responses and extracted fields above.")
    print("If everything looks correct, run the full pipeline:")
    print("  python3 src/generator.py")
    print("  python3 src/extractor.py")
    print("  python3 src/analysis.py")
    print("="*60)


if __name__ == "__main__":
    main()
