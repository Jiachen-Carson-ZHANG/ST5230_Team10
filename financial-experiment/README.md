# Financial News Persona Experiment

**ST5230 Group 10** — Studying how implicit persona labels in system prompts shift LLM outputs on identical financial news risk assessment tasks.

---

## Research Question

To what extent do implicit personas in system prompts inflate output variance, semantic divergence, and decision flip rates of large language models performing identical financial risk assessment tasks?

---

## Experiment Design

| Parameter | Value |
|---|---|
| Input | 20 real financial news articles (Reuters) |
| Personas | 4 — Baseline (no prompt) + 3 implicit role labels |
| Models | 3 — GPT-4o, o3-mini, one small model |
| Samples per condition | 20 (temperature = 0.7) |
| Total generator calls | 20 × 4 × 3 × 20 = **4,800** |
| Extractor | GPT-4o-mini at temperature = 0 |

### Personas

`baseline` remains a true empty control condition. The three non-baseline personas are richer role descriptions injected as system prompts, while the user prompt remains fixed across all conditions.

| ID | System prompt |
|---|---|
| `baseline` | *(empty — no system prompt)* |
| `conservative_officer` | Strictly conservative CRO / compliance auditor focused on capital preservation and legal risk avoidance. |
| `aggressive_hedge_fund` | Highly aggressive hedge fund manager focused on maximum alpha and preemptive action. |
| `neutral_researcher` | Objective equity research analyst focused on balanced, data-driven assessment. |

### Task (user prompt, same for all personas)

Each model is asked for a short free-form report, not JSON. The user prompt injects the target ticker, tells the model to focus only on that holding if multiple companies appear, and requires exactly these three headings:
1. **`[Risk Rating]:`** — overall risk level for the event
2. **`[Strategic Action]:`** — immediate recommended action
3. **`[Justification]:`** — concise explanation, under 200 words total

The extractor is the structured layer. Generator output should stay human-readable and free-form.

---

## Pipeline

```
articles.json          ← normalized Reuters corpus in data/news/
        ↓
[Stage 1] python3 src/generator.py
        ↓
data/raw/responses.jsonl    ← one JSON line per API call, with generator contract version
        ↓
[Stage 2] python3 src/extractor.py
        ↓
data/structured/results.csv ← extracted fields plus extractor version + failure diagnostics
        ↓
[Stage 3] python3 src/analysis.py
        ↓
data/figures/               ← heatmap PNGs for each metric
```

---

## Setup

**1. Copy and fill in config:**
```bash
cp src/config.example.py src/config.py
# Open src/config.py and set SMALL_MODEL_ID to any OpenRouter model ID
# e.g. "meta-llama/llama-3.1-8b-instruct"
```

**2. Set API key:**
```bash
export OPENROUTER_API_KEY=sk-or-...
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Normalize the source corpus:**
```bash
python3 src/prepare_articles.py
```

This reads the workspace-level `news_corpus_final.json` and writes a normalized `data/news/articles.json` with the internal `article_id` / `article_text` schema used by the pipeline.

---

## Running

**Smoke test first — 4 real API calls plus extractor checks:**
```bash
python3 src/test_run.py
```
Review the printed responses, format checks, and extracted mandatory fields. If they look correct, run the full pipeline.
The smoke test treats the 200-word target as a soft band: `<=200` is a pass, `201-240` is a warning, and only `>240` is a hard length failure.

**Full pipeline:**
```bash
python3 src/generator.py   # ~4,800 calls — resumable, safe to interrupt
python3 src/extractor.py   # ~9,600 calls — resumable, safe to interrupt
python3 src/analysis.py    # reads results, saves figures to data/figures/
```

All stages are resumable, but resumability is now version-guarded.
If you change the generator prompt contract or extractor rubric version, start a fresh output file instead of mixing rows across versions.

---

## Extracted Fields (10 dimensions)

| Field | Type | What it measures |
|---|---|---|
| `risk_rating_score` | Integer 1–5 | Overall risk severity (1=bullish, 5=existential crisis) |
| `strategic_action` | Enum | Strong\_Buy / Hold\_Monitor / Reduce\_Exposure / Clear\_Short / Halt\_Compliance |
| `action_urgency` | Enum | Immediate / Short\_Term / Long\_Term |
| `unsupported_financial_claim_flag` | Boolean | Did the model invent specific financial facts not in the article? |
| `compliance_refusal_flag` | Boolean | Did the model refuse to give financial advice (safety guardrails)? |
| `analysis_primary_focus` | Enum | Market\_Sentiment / Fundamentals / Legal\_Regulatory |
| `reasoning_basis` | Enum | News\_Fact\_Driven / Historical\_Analogies / Speculative\_Doom |
| `tone_confidence_level` | Integer 1–5 | Certainty of language (1=uncertain, 5=absolute) |
| `risk_thesis_hook` | String ≤20 words | Core logic summary for human review |
| `output_word_count` | Integer | Exact word count of the raw response |

`results.csv` also includes `extractor_version`, `extraction_error_type`, and `hallucination_error_type` so extractor failures stay diagnosable instead of collapsing silently.

---

## Analysis Metrics

Six independent metrics, each saved as a heatmap PNG:

| Metric | Measures |
|---|---|
| **Flip rate** | % of the 20 samples where derived `strategic_posture` differs from the most common posture |
| **Decision entropy** | Shannon entropy of `strategic_action` distribution across 20 samples |
| **Conservatism score** | Mean `risk_rating_score` per persona × model |
| **Action urgency score** | Mean numeric urgency derived from `action_urgency` (`Immediate`=5, `Short_Term`=2, `Long_Term`=1) |
| **Semantic variance** | Mean pairwise cosine distance of raw response embeddings |
| **Unsupported claim rate** | % of responses where model invented specific financial facts |

Running analysis also exports `data/figures/unsupported_claim_cases.csv` so flagged unsupported-claim rows can be reviewed manually.

---

## File Structure

```
financial-experiment/
├── data/
│   ├── news/articles.json      ← input articles (place here before running)
│   ├── raw/responses.jsonl     ← generator output
│   ├── structured/results.csv  ← extractor output
│   └── figures/                ← analysis heatmaps
├── src/
│   ├── config.example.py       ← copy to config.py and fill in
│   ├── config.py               ← local only, gitignored
│   ├── articles.py             ← article normalization helpers
│   ├── prepare_articles.py     ← converts news_corpus_final.json to articles.json
│   ├── pipeline_records.py     ← shared record builders + version guards
│   ├── prompts.py              ← personas + task instruction
│   ├── generator.py            ← Stage 1
│   ├── extractor.py            ← Stage 2
│   ├── analysis.py             ← Stage 3
│   └── test_run.py             ← smoke test before full run
└── requirements.txt
```

---

## Cost Estimate

| Model | Calls | Est. Cost |
|---|---|---|
| GPT-4o (generator) | 1,600 | ~$12 |
| o3-mini (generator) | 1,600 | ~$8 |
| Small model (generator) | 1,600 | ~$2 |
| GPT-4o-mini (extractor) | 9,600 | ~$3 |
| **Total** | | **~$25** |
