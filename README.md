# CS 568 Food Analysis

This project investigates which recipe attributes correlate with high ratings in the Food.com dataset. It augments recipes with LLM-generated quantitative tags using Google AI Studio's Gemma 3 27B model, then trains a logistic regression model to identify the strongest predictors.

## Project Objective

Determine what features are important to creating a highly-rated dish. We augment the Food.com dataset with LLM-generated qualitative tags and train a Logistic Regression model to analyze correlation coefficients of these features against user ratings.

## Repository Structure

- src/cs568_food_analysis/: package with data prep, LLM augmentation, and analysis modules
- main.py: legacy entrypoint that runs both stages end-to-end
- data/raw/: Food.com datasets (recipes and interactions)
- data/processed/: generated augmented_recipes.csv

## Requirements

- Python 3.14+
- uv (package manager)
- GEMINI_API_KEY in .env for LLM calls

## Setup

More information available in [setup.sh](setup.sh).

```bash
uv sync
cp .env.sample .env
```

Edit .env and set:

```
GEMINI_API_KEY=your_key_here
```

## Data Preparation

```bash
source .venv/bin/activate # This should give you `kaggle` CLI.

mkdir -p data/raw data/processed
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions -p data/raw --unzip
```

## Run the Pipeline

```bash
uv run cs568-prepare
uv run cs568-analyze
```

To keep the old behavior (both stages in one go):

```bash
uv run python main.py
```

Outputs:

- data/processed/augmented_recipes.csv
- console report of feature coefficients (sorted by absolute weight)

## Notes on Rate Limiting

The pipeline batches 10 recipes per request and sleeps 12 seconds between calls to stay under 30 RPM and 15K TPM limits.

## Configuration

Key parameters in src/cs568_food_analysis/config.py:

- BATCH_SIZE: number of recipes per request (default 10)
- SAMPLE_SIZE: number of recipes sampled from the merged dataset
- SLEEP_SECONDS: delay between LLM calls

## License

For CS568 course project use only.
