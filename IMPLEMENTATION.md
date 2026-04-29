# Implementation Details

This document describes the full pipeline used for the CS 568 Food Analysis project, including data preparation, LLM augmentation, and modeling steps.

## 1) Data Preparation

Inputs:

- data/raw/RAW_recipes.csv
- data/raw/RAW_interactions.csv

Steps:

1. Load recipes and keep: recipe_id, name, ingredients, steps.
2. Parse ingredients/steps from stringified lists to readable text.
3. Load interactions and compute average rating per recipe.
4. Merge recipes with average ratings on recipe_id.
5. Sample N recipes (default 20 in config.py for quick runs).

Key functions:

- load_and_prepare_recipes
- load_and_prepare_interactions
- merge_recipes_and_ratings
- sample_recipes

## 2) LLM Augmentation (Gemma 3 27B)

Goal: generate continuous scores (1.0-10.0) for six recipe attributes:

- presentation_quality
- saltiness
- oiliness
- richness
- ingredient_diversity
- complexity_of_preparation

Batching strategy:

- BATCH_SIZE = 10
- Each request includes a JSON array of 10 recipes
- asyncio.sleep(12) after each request

Rate-limiting compliance:

- 10 recipes per call keeps token usage under ~2,500 tokens
- 12-second delay ensures <= 5 calls per minute

Robustness:

- Tenacity exponential backoff for LlmResponseError
- JSON extraction and schema validation
- recipe_id normalized to numeric to avoid merge dtype mismatches

Key functions:

- build_prompt
- generate_batch_features
- augment_with_llm

Output:

- data/processed/augmented_recipes.csv

## 3) Logistic Regression & Feature Analysis

Target variable:

- target = 1 if avg_rating >= 4.5 else 0

Feature matrix:

- X = six LLM-generated attributes
- StandardScaler applied before modeling

Model:

- LogisticRegression(max_iter=200)

Interpretation:

- Coefficients indicate correlation direction and strength
- Sorted by absolute weight for importance ranking

Key function:

- run_logistic_regression

Evaluation additions:

- Baseline features: ingredient_count, steps_length
- Metrics: accuracy and ROC-AUC for LLM vs baseline
- Plots: ROC curve overlay + confusion matrix for LLM model

## 4) How to Run

```bash
uv run cs568-prepare
uv run cs568-analyze
```

Legacy end-to-end run:

```bash
uv run python main.py
```

Outputs:

- data/processed/augmented_recipes.csv
- Console summary of feature coefficients

## 5) Files and Parameters

Package: src/cs568_food_analysis/
Legacy entrypoint: main.py

Important constants:

- BATCH_SIZE
- SAMPLE_SIZE
- SLEEP_SECONDS
- MODEL_NAME
- FEATURE_COLUMNS

## 6) Failure Modes and Troubleshooting

- Missing GEMINI_API_KEY: set in .env
- Merge dtype error on recipe_id: ensure numeric recipe_id in LLM output
- JSON parse errors: check model response format
- Rate limit errors: keep BATCH_SIZE and SLEEP_SECONDS unchanged
