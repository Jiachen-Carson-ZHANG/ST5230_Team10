# Reviewer Reproducibility Guide

This file is a short, reviewer-oriented guide for instructors, TAs, or other readers who want to verify the codebase quickly.

The implementation lives in [financial-experiment/](./financial-experiment). The public repository includes the full source code, prompt definitions, tests, and normalized article inputs. It does **not** include private API credentials or the large generated outputs from the full 4,800-call run.

## What Is Included

- Core pipeline code in [financial-experiment/src/](./financial-experiment/src)
- Prompt templates and persona definitions in [financial-experiment/src/prompts.py](./financial-experiment/src/prompts.py)
- Shared utilities in [financial-experiment/src/articles.py](./financial-experiment/src/articles.py) and [financial-experiment/src/pipeline_records.py](./financial-experiment/src/pipeline_records.py)
- Tests in [financial-experiment/tests/](./financial-experiment/tests)
- Normalized article inputs in [financial-experiment/data/news/articles.json](./financial-experiment/data/news/articles.json) and [financial-experiment/data/news/articles.full.json](./financial-experiment/data/news/articles.full.json)
- A notebook-derived supplementary analysis script at [Analysis.py](./Analysis.py)

## What Is Not Included

- `financial-experiment/src/config.py`
  - This file is intentionally excluded because it is a local credential/config file.
- `financial-experiment/data/raw/responses.jsonl`
- `financial-experiment/data/structured/results.csv`
- generated figures and backup outputs from our own run

Those generated artifacts were omitted because they are large, API-derived outputs. The repo is structured so the pipeline can be rerun cleanly.

## Fastest Verification Path

This path checks that the codebase imports correctly and that the test suite passes. It does **not** require an API key.

```bash
git clone https://github.com/Jiachen-Carson-ZHANG/ST5230_Team10.git
cd ST5230_Team10
python3 -m venv .venv
source .venv/bin/activate
pip install -r financial-experiment/requirements.txt
pytest -q financial-experiment/tests
```

Expected result:

- the tests should pass without needing a local `src/config.py`
- this works because [financial-experiment/src/config_runtime.py](./financial-experiment/src/config_runtime.py) falls back to [financial-experiment/src/config.example.py](./financial-experiment/src/config.example.py) when a local config file is absent

## Full Pipeline Rerun

This path reruns the API-backed experiment. It requires an OpenRouter API key.

```bash
cd financial-experiment
cp src/config.example.py src/config.py
```

Then edit `src/config.py` and set:

- `SMALL_MODEL_ID = "meta-llama/llama-3.3-70b-instruct"`

Then set your API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Then run:

```bash
python3 src/test_run.py
python3 src/generator.py
python3 src/extractor.py
python3 src/analysis.py
```

Notes:

- `src/test_run.py` is the recommended smoke test before the full run
- `src/generator.py`, `src/extractor.py`, and `src/analysis.py` are resumable in the intended workflow
- the normalized article files are already included in `data/news/`, so `src/prepare_articles.py` is **not required** for reviewer reruns

## Files and Stages

The main scripts are:

- [financial-experiment/src/generator.py](./financial-experiment/src/generator.py)
- [financial-experiment/src/extractor.py](./financial-experiment/src/extractor.py)
- [financial-experiment/src/analysis.py](./financial-experiment/src/analysis.py)

Additional helper scripts:

- [financial-experiment/src/prepare_articles.py](./financial-experiment/src/prepare_articles.py)
  - used to regenerate normalized articles from a separate source corpus file that is not shipped in this public repo
- [financial-experiment/src/test_run.py](./financial-experiment/src/test_run.py)
  - smoke test for a small API-backed end-to-end check

## Structured Outputs and Main Fields

The extraction layer produces structured fields including:

- `risk_rating_score`
- `strategic_action`
- `action_urgency`
- `analysis_primary_focus`
- `unsupported_financial_claim_flag`
- `tone_confidence_level`
- `output_word_count`

These fields feed the reported downstream analyses.

## Model Configuration Used in the Study

- Generator models:
  - `openai/gpt-4o`
  - `openai/o3-mini`
  - `meta-llama/llama-3.3-70b-instruct`
- Extractor model:
  - `openai/gpt-4o-mini`

Sampling details:

- GPT-4o and Llama 3.3 were run with `temperature = 0.7`
- o3-mini used its provider-default reasoning-model parameters
- the extractor used `temperature = 0`

Contract versions:

- generator version: `2026-03-27-freeform-generator-v1`
- extractor version: `2026-03-28-freeform-extractor-v4`

## Practical Reproducibility Note

Because the generator stage uses external APIs and no explicit random seed, bit-for-bit regeneration of the original raw outputs is not guaranteed. Reproducibility in this project therefore means:

- the code, prompts, and schema logic are available for inspection
- the test suite verifies the key parsing and metric logic
- the experiment can be rerun with the same pipeline structure, model IDs, and prompt contracts

That is the intended standard for review of this repository.
