# CS 568 Food Analysis

This project investigates which recipe attributes correlate with high ratings in the Food.com dataset. It augments recipes with LLM-generated quantitative tags using Google AI Studio's Gemma 3 27B model, then trains a logistic regression model to identify the strongest predictors.

## Project Objective

Determine what features are important to creating a highly-rated dish. We augment the Food.com dataset with LLM-generated qualitative tags and train a Logistic Regression model to analyze correlation coefficients of these features against user ratings.

## Repository Structure

- src/cs568_food_analysis/: package with data prep, LLM augmentation, and analysis modules
- src/recipe_optimizer_demo.py: standalone demo script for the final presentation
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

To save the radar plot instead of showing it:

```bash
uv run cs568-analyze --plot-path data/processed/feature_radar.png
```

Evaluation artifacts (ROC curve + confusion matrix) are saved to the processed
directory by default:

```bash
uv run cs568-analyze --eval-output-dir data/processed
```

Run the demo script:

```bash
uv run python src/recipe_optimizer_demo.py
```

To keep the old behavior (both stages in one go):

```bash
uv run python main.py
```

Outputs:

- data/processed/augmented_recipes.csv
- data/processed/roc_curve.png
- data/processed/confusion_matrix.png
- console report of feature coefficients (sorted by absolute weight)

Note: avg_rating is computed during analysis from raw interactions; it is no longer
stored in the processed CSV.
## Experiments

How everything is calculated:

- Data cleaning: remove unreliable users (std=0.0 with multiple reviews), then keep
     recipes with at least `MIN_REVIEWS` reviews.
- LLM augmentation: each recipe gets six 1.0-10.0 aspect scores from Gemma 3.
- Target: compute `avg_rating` from raw interactions, then apply a median split
     (rating > median => 1, else 0).
- Baseline: a single numeric feature, `ingredient_count` (number of comma-separated
     ingredients).
- Evaluation: compare LLM features vs baseline `ingredient_count` with a
     RandomForest classifier; report Accuracy + ROC-AUC, plus ROC curve and confusion
     matrix plots.

## Results

Analysis results are printed to the console after `uv run cs568-analyze` and visualized
in the saved plots below.



**Analysis Results**

Out of 1000 recipes sampled, there are 228 Effective recipes after data cleaning.

| Feature | Coefficient |
| --- | --- |
| complexity_of_preparation | -0.160115 |
| saltiness | -0.151902 |
| richness | -0.136621 |
| presentation_quality | 0.135311 |
| oiliness | 0.133328 |
| ingredient_diversity | -0.076834 |

| Metric | Accuracy | ROC-AUC |
| --- | --- |
| LLM (ours) | **0.630** |**0.662**|
| Baseline | 0.587 | 0.609 |

**Analysis Plots**
- ![Feature Radar](data/processed/feature_radar.png)
- ![ROC Curve](data/processed/roc_curve.png)
- ![Confusion Matrix](data/processed/confusion_matrix.png)



**Demo Results**
The demo script prints a before/after comparison for the baseline and optimized recipes,
including feature shifts and predicted win probabilities:

```text
=========================================
[1] THE BASELINE RECIPE
Name: Heavy Salted Cheese & Cream Casserole Mash
Ingredients: 3 cups heavy cream, 2 cups processed cheese, 1 cup butter, 4 tbsp salt, 2 lbs ground beef, canned potatoes.
Steps: Dump everything into a slow cooker. Cook for 8 hours until it becomes a homogenous grey sludge. Add extra salt to taste. Serve in a large messy bowl.

[2] THE OPTIMIZED RECIPE (Air Fryer/Pressure Cooker Modernization)
Name: Savory Beef & Potato Mash with Herbs
Ingredients: 2 lbs lean ground beef, 2 lbs Yukon Gold potatoes (peeled and quartered), 1 cup beef broth (low sodium), 1/2 cup plain Greek yogurt, 2 tbsp olive oil, 1 tbsp fresh rosemary (chopped), 1 tsp black pepper, 1/2 tsp garlic powder
Steps: 1. Brown ground beef in a large pot with olive oil over medium-high heat. Drain any excess fat. 2. Add potatoes and beef broth to the pot. Bring to a boil, then reduce heat and simmer for 15-20 minutes, or until potatoes are tender. 3. Drain any remaining liquid. Mash potatoes and beef together until well combined. 4. Stir in Greek yogurt, rosemary, pepper, and garlic powder. Mix well. 5. To plate: Spoon mash into shallow bowls. Drizzle with a tiny amount of olive oil and garnish with a sprig of fresh rosemary.

[3] FEATURE SHIFT ANALYSIS
- presentation_quality: 1.5 -> 5.5
- saltiness: 9.0 -> 3.0
- oiliness: 8.5 -> 3.0
- richness: 10.0 -> 6.0
- ingredient_diversity: 4.0 -> 7.0
- complexity_of_preparation: 1.5 -> 5.0
[4] PREDICTION SHOWDOWN
- Baseline Win Probability: 27.7%
- Optimized Win Probability: 56.5%
=========================================
```

## Notes on Rate Limiting

The pipeline batches 10 recipes per request and sleeps 12 seconds between calls to stay under 30 RPM and 15K TPM limits.

## Configuration

Key parameters in src/cs568_food_analysis/config.py:

- BATCH_SIZE: number of recipes per request (default 10)
- SAMPLE_SIZE: number of recipes sampled from the filtered dataset
- SLEEP_SECONDS: delay between LLM calls
- MIN_REVIEWS: minimum review count for recipe eligibility

## License

For CS568 course project use only.
