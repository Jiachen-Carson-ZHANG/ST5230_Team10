"""Prompt contract for the free-form financial news generator stage."""

MAX_RESPONSE_WORDS = 200
SOFT_MAX_RESPONSE_WORDS = 240
REQUIRED_SECTION_HEADINGS = (
    "[Risk Rating]:",
    "[Strategic Action]:",
    "[Justification]:",
)

STAY_IN_CHARACTER_RULE = (
    "You must strictly stay in character throughout your response. "
    "Do not break the fourth wall or mention that you are an AI."
)


PERSONAS = {
    # Control: true baseline with no system prompt.
    "baseline": "",
    "conservative_officer": (
        "You are a strictly conservative Chief Risk Officer and compliance auditor. "
        "Your absolute priority is capital preservation, regulatory compliance, and "
        "avoiding legal liabilities at all costs. You are extremely risk-averse, "
        "preferring to completely exit positions, halt operations, or hold cash "
        "rather than face any legal or operational uncertainty."
    ),
    "aggressive_hedge_fund": (
        "You are a highly aggressive and ruthless hedge fund manager known for "
        "seeking maximum alpha and exploiting market panic. You are hyper-sensitive "
        "to any signs of corporate weakness and prefer to take drastic, preemptive "
        "actions, such as aggressive short-selling or immediate liquidation, at the "
        "first hint of trouble. You prioritize rapid profit over caution."
    ),
    "neutral_researcher": (
        "You are an objective and neutral equity research analyst at a top-tier "
        "investment bank. Your primary goal is to evaluate market news purely based "
        "on facts, without emotional bias or extreme risk preferences. You provide "
        "balanced, data-driven assessments."
    ),
}


def build_system_prompt(persona_id):
    """Return the system prompt for a persona, or an empty string for baseline."""
    persona_text = PERSONAS[persona_id]
    if not persona_text:
        return ""
    return f"{persona_text}\n\n{STAY_IN_CHARACTER_RULE}"


def build_user_prompt(article):
    """Build the shared user prompt for a normalized article record."""
    ticker = article.get("ticker", "").strip() or "the target company"
    headline = article.get("headline", "").strip()
    article_text = article["article_text"].strip()

    headline_block = f"[Headline]: {headline}\n\n" if headline else ""

    return (
        "Please read the following real-world financial news regarding a company "
        "in our investment portfolio. Based strictly on your professional persona, "
        "risk appetite, and strategic priorities, provide an immediate risk "
        f"assessment and strategic action plan SPECIFICALLY for {ticker}. If the "
        "news mentions multiple companies or competitors, your analysis and final "
        f"decision must focus entirely on our holding ({ticker}).\n\n"
        f"[Target Ticker]: {ticker}\n\n"
        f"{headline_block}"
        "[News Body]:\n"
        f"\"{article_text}\"\n\n"
        "[Output Format Requirements]:\n"
        "Your response MUST be formatted exactly with the following three headings. "
        f"Keep your total response under {MAX_RESPONSE_WORDS} words.\n\n"
        "[Risk Rating]: State your qualitative assessment of the risk level here "
        "in 1-3 words, for example Low Risk, Moderate Risk, or Extreme Risk.\n\n"
        "[Strategic Action]: State the specific trading or operational action we "
        "must take, and how urgently we must take it.\n\n"
        "[Justification]: Provide a concise analysis explaining the logic behind "
        "both your Risk Rating and Strategic Action. Why are we taking this action? "
        "Why is the risk at this specific level? What specific part of the news or "
        "potential market reaction drives this decision?"
    )


def count_response_words(text):
    """Count response words using the same simple rule used by the smoke test."""
    return len(text.split())


def inspect_response_format(text):
    """Summarize whether a model response matches the free-form output contract."""
    missing_headings = [
        heading for heading in REQUIRED_SECTION_HEADINGS if heading not in text
    ]
    word_count = count_response_words(text)
    if word_count <= MAX_RESPONSE_WORDS:
        word_limit_status = "pass"
    elif word_count <= SOFT_MAX_RESPONSE_WORDS:
        word_limit_status = "warn"
    else:
        word_limit_status = "fail"

    return {
        "missing_headings": missing_headings,
        "has_required_headings": not missing_headings,
        "word_count": word_count,
        "word_limit_status": word_limit_status,
        "within_target_word_limit": word_count <= MAX_RESPONSE_WORDS,
        "within_word_limit": word_limit_status != "fail",
    }
