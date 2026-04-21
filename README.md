# ST5230 Team 10 — Financial News Persona Experiment

This repository contains the implementation for our ST5230 project on how persona prompting changes LLM behavior in financial-news risk assessment.

The main implementation lives in [financial-experiment/](./financial-experiment).

## Repository Overview

- [financial-experiment/src/](./financial-experiment/src)
  - generator, extractor, analysis, prompt, and utility code
- [financial-experiment/tests/](./financial-experiment/tests)
  - reproducibility and regression tests
- [Analysis.py](./Analysis.py)
  - supplementary script derived from the final analysis notebooks
- [REVIEWER_README.md](./REVIEWER_README.md)
  - detailed TA/professor-focused reproducibility guide

## For Reviewers

The public repository includes:

- source code
- prompt templates
- tests
- normalized article inputs

The public repository does **not** include:

- private API credentials
- the full generated raw outputs
- the full extracted `results.csv`

This means there are two practical ways to review the project.

### 1. Fast Verification

This is the recommended path for a professor or TA who wants to confirm that the codebase is complete and runnable without spending API cost.

```bash
git clone https://github.com/Jiachen-Carson-ZHANG/ST5230_Team10.git
cd ST5230_Team10
python3 -m venv .venv
source .venv/bin/activate
pip install -r financial-experiment/requirements.txt
pytest -q financial-experiment/tests
```

Expected result:

- the test suite should pass without a local `financial-experiment/src/config.py`
- the repo supports this through [financial-experiment/src/config_runtime.py](./financial-experiment/src/config_runtime.py), which falls back to [financial-experiment/src/config.example.py](./financial-experiment/src/config.example.py)

### 2. Full Pipeline Rerun

This path reruns the API-backed experiment and therefore requires an OpenRouter API key.

```bash
cd financial-experiment
cp src/config.example.py src/config.py
```

Then edit `src/config.py` and set:

- `SMALL_MODEL_ID = "meta-llama/llama-3.3-70b-instruct"`

Then export your API key:

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

The normalized article inputs are already included in [financial-experiment/data/news/](./financial-experiment/data/news), so `src/prepare_articles.py` is not required for reviewer reruns.

## Experimental Configuration

- Generator models:
  - `openai/gpt-4o`
  - `openai/o3-mini`
  - `meta-llama/llama-3.3-70b-instruct`
- Extractor model:
  - `openai/gpt-4o-mini`
- Generator contract version:
  - `2026-03-27-freeform-generator-v1`
- Extractor contract version:
  - `2026-03-28-freeform-extractor-v4`

Sampling configuration:

- GPT-4o and Llama 3.3 used `temperature = 0.7`
- o3-mini used provider-default reasoning-model parameters
- the extractor used `temperature = 0`

## Reproducibility Note

Because the generator stage uses external APIs and no explicit random seed, bit-for-bit regeneration of the original raw outputs is not guaranteed. In this project, reproducibility means:

- the implementation is fully inspectable
- the tests verify the parsing and metric logic
- the pipeline can be rerun with the same prompt contracts, model IDs, and evaluation structure

For the fuller reviewer-oriented explanation, see [REVIEWER_README.md](./REVIEWER_README.md).
