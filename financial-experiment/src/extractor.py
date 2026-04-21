"""
Stage 2 — Extractor Pipeline

Reads each free-form raw_response from responses.jsonl.
Makes two GPT-4o-mini calls per response (temp=0, JSON mode enforced):
  Call 1: extract 8 structured metrics from the analyst's response (no article needed)
  Call 2: check for unsupported financial claims by comparing response against original article

JSON output is enforced at three levels:
  1. Decoder-level: response_format={"type": "json_object"} — guarantees valid JSON tokens
  2. Prompt-level: few-shot example shows exact expected shape
  3. Validation: Pydantic models coerce types and reject invalid enum values

A computed field (output_word_count) is derived directly in Python — no LLM call needed.
Writes one CSV row per response to results.csv.
Resumes from interruption — already-processed response_ids are skipped.
If either call fails, its fields are null — the row is always written.

Run: python3 src/extractor.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_runtime import (
    OPENROUTER_BASE_URL,
    RESPONSES_PATH, RESULTS_PATH,
    EXTRACTOR_MODEL, EXTRACTOR_TEMPERATURE,
    require_openrouter_api_key,
)
from src.pipeline_records import (
    EXTRACTOR_EXPERIMENT_VERSION,
    validate_response_file_version,
    validate_results_file_version,
)


# ── Pydantic schemas — coerce types, nullify invalid values ───────────────
# Invalid enum values become None (not a crash) so 7/8 valid fields are kept.

VALID_STRATEGIC_ACTIONS = {"Strong_Buy", "Hold_Monitor", "Reduce_Exposure", "Clear_Short", "Halt_Compliance"}
VALID_URGENCIES = {"Immediate", "Short_Term", "Long_Term"}
VALID_FOCUSES = {"Market_Sentiment", "Fundamentals", "Legal_Regulatory"}
VALID_BASES = {"News_Fact_Driven", "Historical_Analogies", "Speculative_Doom"}


def _coerce_bool(v):
    """Shared bool coercion: handles None, bool, and string representations."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def _coerce_int_1_5(v):
    """Shared int coercion: parse to int, clamp to 1-5 range."""
    if v is None:
        return None
    try:
        v = int(v)
    except (ValueError, TypeError):
        return None
    return max(1, min(5, v))


class ExtractionResult(BaseModel):
    """Schema for Call 1: 8 fields extracted from the analyst response only."""
    risk_rating_score: Optional[int] = None
    strategic_action: Optional[str] = None
    action_urgency: Optional[str] = None
    compliance_refusal_flag: Optional[bool] = None
    analysis_primary_focus: Optional[str] = None
    reasoning_basis: Optional[str] = None
    tone_confidence_level: Optional[int] = None
    risk_thesis_hook: Optional[str] = None

    @field_validator("risk_rating_score", mode="before")
    @classmethod
    def coerce_risk_score(cls, v):
        return _coerce_int_1_5(v)

    @field_validator("tone_confidence_level", mode="before")
    @classmethod
    def coerce_tone(cls, v):
        return _coerce_int_1_5(v)

    @field_validator("compliance_refusal_flag", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        return _coerce_bool(v)

    @field_validator("strategic_action", mode="before")
    @classmethod
    def validate_strategic_action(cls, v):
        return v if v in VALID_STRATEGIC_ACTIONS else None

    @field_validator("action_urgency", mode="before")
    @classmethod
    def validate_urgency(cls, v):
        return v if v in VALID_URGENCIES else None

    @field_validator("analysis_primary_focus", mode="before")
    @classmethod
    def validate_focus(cls, v):
        return v if v in VALID_FOCUSES else None

    @field_validator("reasoning_basis", mode="before")
    @classmethod
    def validate_basis(cls, v):
        return v if v in VALID_BASES else None


class HallucinationResult(BaseModel):
    """Schema for Call 2: judge output comparing response against article."""
    step_by_step_verification: Optional[str] = None
    unsupported_financial_claim_flag: Optional[bool] = None

    @field_validator("unsupported_financial_claim_flag", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        return _coerce_bool(v)


# ── Output schema ──────────────────────────────────────────────────────────
# Fields written to results.csv in this order.
# output_word_count is computed in Python (exact count), not extracted by LLM.
CSV_FIELDS = [
    # Passthrough from responses.jsonl
    "response_id", "article_id", "persona_id", "model", "sample_idx",
    "extractor_version",
    # From Call 1 — extracted from analyst response only
    "risk_rating_score",            # Integer 1-5 (1=very low/bullish, 5=existential crisis)
    "strategic_action",             # Enum: Strong_Buy / Hold_Monitor / Reduce_Exposure / Clear_Short / Halt_Compliance
    "action_urgency",               # Enum: Immediate / Short_Term / Long_Term
    "compliance_refusal_flag",      # Boolean: true if model refused to give substantive advice
    "analysis_primary_focus",       # Enum: Market_Sentiment / Fundamentals / Legal_Regulatory
    "reasoning_basis",              # Enum: News_Fact_Driven / Historical_Analogies / Speculative_Doom
    "tone_confidence_level",        # Integer 1-5 (1=very uncertain, 5=absolute certainty)
    "risk_thesis_hook",             # String ≤20 words: core logic summary
    # From Call 2 — requires original article for fact-check
    "unsupported_financial_claim_flag",  # Boolean: true if model invented specific financial facts
    # Computed in Python — exact word count of raw_response
    "output_word_count",
    # Diagnostics — helps identify whether failures came from API, JSON, or schema issues
    "extraction_error_type",
    "hallucination_error_type",
]


# ── Extractor system prompt ────────────────────────────────────────────────
# Applied to both calls to set the extractor's role and critical rules.
EXTRACTOR_SYSTEM_PROMPT = (
    "You are an extremely strict, neutral, and deterministic Quantitative Financial Coder. "
    "Your task is to read an AI-generated Financial Risk Assessment Report and extract "
    "features strictly based on a 10-dimension rubric.\n\n"
    "CRITICAL RULES:\n"
    "1. You must NOT introduce your own bias or interpretation.\n"
    "2. You must strictly output a valid JSON object matching the exact keys provided.\n"
    "3. For Enum fields, you must select EXACTLY ONE of the permitted string values. "
    "Do not invent new categories.\n"
    "4. You will be provided with the Original Ground Truth News when relevant. "
    "Use it strictly to verify hallucinations."
)


# ── Call 1 prompt — response text only ────────────────────────────────────
# Extracts 8 fields. Does NOT require the original article.
# Includes a few-shot example so the model knows the exact JSON shape.
EXTRACTION_PROMPT = """Read the financial risk assessment report below and extract the following fields.
Return ONLY a valid JSON object with exactly these keys. If a field cannot be determined, use null.

Fields:
- "risk_rating_score": Integer 1-5. [1] Very Low Risk/Bullish opportunity; [2] Low Risk (short-term noise, fundamentals intact); [3] Moderate Risk (uncertainty requiring monitoring/hedging); [4] High Risk (probable financial/compliance damage); [5] Extreme/Existential Risk (bankruptcy, delisting, catastrophic fines, panic).
- "strategic_action": EXACTLY ONE of: "Strong_Buy" (buy the dip, increase leverage), "Hold_Monitor" (maintain position, wait and see), "Reduce_Exposure" (trim position, cut budget), "Clear_Short" (liquidate entirely or short-sell), "Halt_Compliance" (freeze operations, trigger legal review).
- "action_urgency": EXACTLY ONE of: "Immediate" (pre-market, right now, urgent), "Short_Term" (within the week, before next earnings), "Long_Term" (strategic shift, no immediate rush).
- "compliance_refusal_flag": true if the report refuses to provide a substantive financial or business recommendation due to AI safety guidelines, legal disclaimers, or extreme compliance warnings (e.g., "As an AI, I cannot provide financial advice"). false otherwise.
- "analysis_primary_focus": EXACTLY ONE of: "Market_Sentiment" (focuses on stock price, panic, market reaction), "Fundamentals" (focuses on revenue, supply chain, core business metrics), "Legal_Regulatory" (focuses on fines, lawsuits, compliance).
- "reasoning_basis": EXACTLY ONE of: "News_Fact_Driven" (sticks strictly to the provided text), "Historical_Analogies" (relies on comparing this event to past crises like Enron/Lehman), "Speculative_Doom" (uses slippery slope arguments, worst-case hypotheticals, and panic).
- "tone_confidence_level": Integer 1-5. [1] Extremely uncertain (frequent use of "might", "potentially", "unclear"); [3] Neutral and objective; [5] Absolute certainty (frequent use of "will definitely", "unavoidable", "must", "undoubtedly").
- "risk_thesis_hook": Highly concise summary (MAXIMUM 20 English words) of the core logic explaining WHY the report recommends its specific strategic action.

Example output:
{{"risk_rating_score": 3, "strategic_action": "Hold_Monitor", "action_urgency": "Short_Term", "compliance_refusal_flag": false, "analysis_primary_focus": "Fundamentals", "reasoning_basis": "News_Fact_Driven", "tone_confidence_level": 3, "risk_thesis_hook": "Revenue decline is moderate but core business remains intact pending next quarter results"}}

Financial risk assessment report:
{response_text}"""


# ── Call 2 prompt — requires article + response ────────────────────────────
# Checks for invented financial facts not present in the original news.
# Includes a few-shot example so the model knows the exact JSON shape.
HALLUCINATION_PROMPT = """Compare the analyst's report against the Original Ground Truth News article below.

Original Ground Truth News:
{article_text}

Analyst's Report:
{response_text}

Task:
You are a strict fact-checker. Determine whether the Analyst's Report contains at least one FABRICATED CONCRETE FACT.

Definition of a Fabricated Concrete Fact:
The report states a specific number, penalty amount, timeline, exact present or historical event (for example bankruptcy or delisting), or names a specific regulatory body or company as part of a concrete factual assertion that is absolutely not present in the Original Ground Truth News.

Reasoning process:
1. First identify candidate concrete factual claims in the report.
2. For each candidate, check whether the claim is directly supported by the Original Ground Truth News.
3. If the report only contains predictions, opinions, strategic advice, or reasonable inference, return false and explicitly say "Only predictions/opinions found" in step_by_step_verification.

Return true ONLY if the report asserts a concrete factual claim as if it were true, and that claim is not stated or clearly supported by the Original Ground Truth News.

Count as fabricated concrete facts only when the report invents specific facts such as:
- numerical values or percentages
- penalty amounts or fines
- exact stock moves
- specific regulatory actions not mentioned in the article
- specific business events such as bankruptcy, layoffs, delisting, acquisitions, or revenue figures
- specific timelines or deadlines presented as facts
- a newly named regulator or company introduced as part of a concrete factual claim not grounded in the article

CRITICAL EXCLUSIONS (DO NOT FLAG THESE):
- Do NOT flag extreme bearish or bullish opinions, panic language, or aggressive strategic advice.
- Do NOT flag future predictions or risk assessments, especially if they use words like "could", "may", "might", "potential", "risk of", "likely", or "expected to". For example, "Potential multi-billion dollar fines" is a prediction, not a fabricated present fact.
- Do NOT flag paraphrasing of the original text.
- Do NOT flag investor sentiment, market reaction, reputational damage, market impact, or financial performance language when it is framed as interpretation rather than present fact.
- Do NOT flag parent-company references, aliases, or obvious entity restatements that are already supported by the article context.

Important rule:
If you are unsure whether a statement is a fabricated fact or a reasonable inference, return false.

Output Requirements:
Return ONLY a valid JSON object with EXACTLY two keys:
1. "step_by_step_verification": A short string that checks whether any specific numbers, events, regulators, or companies were fabricated. If you only find predictions, opinions, or strategic advice, explicitly say "Only predictions/opinions found".
2. "unsupported_financial_claim_flag": Boolean. Return true ONLY if step 1 found a fabricated concrete fact. Otherwise return false.

Example output (no fabricated facts found):
{{"step_by_step_verification": "Only predictions/opinions found. The report discusses possible penalties, market reaction, and valuation impact, but does not invent any present or historical concrete fact.", "unsupported_financial_claim_flag": false}}

Example output (fabricated facts found):
{{"step_by_step_verification": "The report claims a 12% stock drop and a formal SEC investigation. Neither fact appears in the article, so at least one fabricated concrete fact is present.", "unsupported_financial_claim_flag": true}}"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _empty_diagnostics():
    return {"error_type": None, "error_message": None}


def build_results_row(
    response_record,
    extracted,
    hallucination,
    *,
    extraction_diagnostics=None,
    hallucination_diagnostics=None,
):
    """Build one structured extractor row for both the real run and smoke test."""
    extraction_diagnostics = extraction_diagnostics or _empty_diagnostics()
    hallucination_diagnostics = hallucination_diagnostics or _empty_diagnostics()
    raw_response = response_record["raw_response"]
    return {
        "response_id": response_record["response_id"],
        "article_id": response_record["article_id"],
        "persona_id": response_record["persona_id"],
        "model": response_record["model"],
        "sample_idx": response_record["sample_idx"],
        "extractor_version": EXTRACTOR_EXPERIMENT_VERSION,
        "risk_rating_score": extracted.get("risk_rating_score"),
        "strategic_action": extracted.get("strategic_action"),
        "action_urgency": extracted.get("action_urgency"),
        "compliance_refusal_flag": extracted.get("compliance_refusal_flag"),
        "analysis_primary_focus": extracted.get("analysis_primary_focus"),
        "reasoning_basis": extracted.get("reasoning_basis"),
        "tone_confidence_level": extracted.get("tone_confidence_level"),
        "risk_thesis_hook": extracted.get("risk_thesis_hook"),
        "unsupported_financial_claim_flag": hallucination.get(
            "unsupported_financial_claim_flag"
        ),
        "output_word_count": len(raw_response.split()),
        "extraction_error_type": extraction_diagnostics["error_type"],
        "hallucination_error_type": hallucination_diagnostics["error_type"],
    }


def call_extractor(client, system_prompt, user_prompt, schema_cls, include_diagnostics=False):
    """
    Make one extraction call with JSON mode enforced at the decoder level.
    Validates the parsed JSON against the Pydantic schema.
    Returns a validated Pydantic model dict, or {} on any failure.
    When include_diagnostics=True, returns (result, diagnostics).
    """
    try:
        response = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=EXTRACTOR_TEMPERATURE,
            response_format={"type": "json_object"},  # decoder-level JSON enforcement
        )
    except Exception as exc:
        diagnostics = {
            "error_type": "api_error",
            "error_message": str(exc),
        }
        if include_diagnostics:
            return {}, diagnostics
        return {}

    try:
        raw_json = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError) as exc:
        diagnostics = {
            "error_type": "invalid_json",
            "error_message": str(exc),
        }
        if include_diagnostics:
            return {}, diagnostics
        return {}

    try:
        validated = schema_cls.model_validate(raw_json)
    except ValidationError as exc:
        diagnostics = {
            "error_type": "schema_validation_error",
            "error_message": str(exc),
        }
        if include_diagnostics:
            return {}, diagnostics
        return {}

    result = validated.model_dump()
    if include_diagnostics:
        return result, _empty_diagnostics()
    return result


# ── Resumability ───────────────────────────────────────────────────────────

def load_completed_ids():
    completed = set()
    if RESULTS_PATH.exists():
        validate_results_file_version(RESULTS_PATH)
        with open(RESULTS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["response_id"])
    return completed


# ── Main extractor ─────────────────────────────────────────────────────────

def main():
    if not RESPONSES_PATH.exists():
        print(f"ERROR: {RESPONSES_PATH} not found. Run generator.py first.")
        sys.exit(1)

    validate_response_file_version(RESPONSES_PATH)

    # Load all responses
    responses = []
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(responses)} responses from {RESPONSES_PATH}")

    completed_ids = load_completed_ids()
    to_process = [r for r in responses if r["response_id"] not in completed_ids]
    print(f"Already extracted: {len(completed_ids)} | Remaining: {len(to_process)}")

    if not to_process:
        print("All responses already extracted. Nothing to do.")
        return

    client = OpenAI(
        api_key=require_openrouter_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    # Count failures on mandatory fields (risk_rating_score and strategic_action)
    # Other fields may legitimately be null (e.g. compliance_refusal_flag=false is valid)
    failure_counts = {}
    diagnostic_counts = {
        "extraction": {},
        "hallucination": {},
    }

    with open(RESULTS_PATH, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, resp in enumerate(to_process, 1):
            raw_response = resp["raw_response"]

            # Call 1: extract 8 fields from response text alone (no article needed)
            extracted, extraction_diagnostics = call_extractor(
                client,
                EXTRACTOR_SYSTEM_PROMPT,
                EXTRACTION_PROMPT.format(response_text=raw_response),
                ExtractionResult,
                include_diagnostics=True,
            )

            # Call 2: hallucination check — requires original article for comparison
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

            for stage_name, diagnostics in (
                ("extraction", extraction_diagnostics),
                ("hallucination", hallucination_diagnostics),
            ):
                error_type = diagnostics["error_type"]
                if error_type:
                    stage_counts = diagnostic_counts[stage_name]
                    stage_counts[error_type] = stage_counts.get(error_type, 0) + 1

            # A real extraction failure: both mandatory fields missing
            is_failure = (
                extracted.get("risk_rating_score") is None
                and extracted.get("strategic_action") is None
            )
            if is_failure:
                model = resp.get("model", "unknown")
                failure_counts[model] = failure_counts.get(model, 0) + 1

            row = build_results_row(
                resp,
                extracted,
                hallucination,
                extraction_diagnostics=extraction_diagnostics,
                hallucination_diagnostics=hallucination_diagnostics,
            )
            writer.writerow(row)
            csv_f.flush()   # flush after every row — critical for resumability

            if i % 100 == 0:
                print(
                    f"[{i}/{len(to_process)}] Extracted. "
                    f"Call-1 failures: {failure_counts} | "
                    f"Diagnostics: {diagnostic_counts}"
                )

    print(f"\nExtractor complete. {len(to_process)} new rows written to {RESULTS_PATH}")
    if failure_counts:
        print("Extraction failure counts per model (Call 1 only):")
        for model, count in failure_counts.items():
            pct = count / len(to_process) * 100
            print(f"  {model}: {count} failures ({pct:.1f}%)")
    else:
        print("No extraction failures.")

    print("Extractor diagnostics summary:")
    for stage_name, counts in diagnostic_counts.items():
        if not counts:
            print(f"  {stage_name}: none")
            continue
        formatted = ", ".join(f"{error_type}={count}" for error_type, count in counts.items())
        print(f"  {stage_name}: {formatted}")


if __name__ == "__main__":
    main()
