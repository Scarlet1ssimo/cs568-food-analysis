"""Standalone Recipe Optimizer demo script for the final presentation."""

import asyncio
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cs568_food_analysis.config import (
    FEATURE_COLUMNS,
    MIN_REVIEWS,
    PROCESSED_DIR,
    RAW_DIR,
    get_interactions_path,
    get_output_path,
)
from cs568_food_analysis.data_prep import load_and_prepare_interactions

MODEL_NAME = "gemma-3-27b-it"
MODEL_PATH = os.path.join(PROCESSED_DIR, "combined_logreg_model.pkl")
SCALER_PATH = os.path.join(PROCESSED_DIR, "combined_scaler.pkl")


class LlmResponseError(RuntimeError):
    pass


@dataclass
class Recipe:
    recipe_id: int
    name: str
    ingredients: str
    steps: str


BASELINE_RECIPE = Recipe(
    recipe_id=1,
    name="Heavy Salted Cheese & Cream Casserole Mash",
    ingredients=(
        "3 cups heavy cream, 2 cups processed cheese, 1 cup butter, "
        "4 tbsp salt, 2 lbs ground beef, canned potatoes."
    ),
    steps=(
        "Dump everything into a slow cooker. Cook for 8 hours until it becomes a "
        "homogenous grey sludge. Add extra salt to taste. Serve in a large messy bowl."
    ),
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise LlmResponseError("Missing JSON object in response")
    return json.loads(text[start : end + 1])


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise LlmResponseError("Missing JSON array in response")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise LlmResponseError("Expected a JSON array in response")
    return parsed


@retry(
    retry=retry_if_exception_type(LlmResponseError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def optimize_recipe(client: genai.Client, baseline: Recipe) -> Recipe:
    prompt = f"""You are a professional culinary R&D chef redesigning a bad recipe.

Redesign the recipe below by strictly following these rules, aligned to feature weights:
1) Reduce complexity_of_preparation: simplify steps and avoid multi-stage techniques.
2) Reduce saltiness and richness using lighter, fresher alternatives.
3) Reduce ingredient_diversity: use fewer, cleaner ingredients (avoid long lists).
4) Boost presentation_quality with a dedicated final plating step.
5) Maintain or slightly increase oiliness using a small amount of healthy oil.

Return ONLY valid JSON in this format (no markdown):
{{
  "name": "...",
  "ingredients": "...",
  "steps": "..."
}}

Baseline Recipe:
Name: {baseline.name}
Ingredients: {baseline.ingredients}
Steps: {baseline.steps}
"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    if not response or not response.text:
        raise LlmResponseError("Empty response from model")

    parsed = _extract_json_object(response.text)
    for key in ("name", "ingredients", "steps"):
        if key not in parsed:
            raise LlmResponseError(f"Missing key in optimized recipe: {key}")

    return Recipe(
        recipe_id=2,
        name=str(parsed["name"]),
        ingredients=str(parsed["ingredients"]),
        steps=str(parsed["steps"]),
    )


def build_scoring_prompt(recipes: List[Recipe]) -> str:
    payload = [recipe.__dict__ for recipe in recipes]
    recipes_json = json.dumps(payload, ensure_ascii=True)
    return f"""You are an expert culinary data scientist extracting quantitative features from recipe texts.
Analyze the following {len(recipes)} recipes and assign continuous scores (from 1.0 to 10.0, allowing one decimal place) for six specific attributes.

CRITICAL INSTRUCTION: Maintain an absolute, objective scale across all evaluations. Do NOT score these recipes relative to each other. Use the following strict rubrics as your absolute anchors:

1. presentation_quality:
   - 1.0: Unappealing mush/slop, visually completely homogenous.
   - 5.0: Standard home-cooked meal, basic plating.
   - 10.0: Michelin-star level aesthetic, vibrant colors, highly structured distinct components.

2. saltiness:
   - 1.0: Zero added sodium, inherently bland raw ingredients.
   - 5.0: Moderately seasoned (e.g., a standard pinch of salt per serving).
   - 10.0: Intensely salty, heavily reliant on brine, soy sauce, or cured meats.

3. oiliness:
   - 1.0: Completely fat-free (e.g., boiled, steamed, or raw veggies).
   - 5.0: Standard pan-fried or roasted with moderate fats.
   - 10.0: Deep-fried or built on a heavy butter/cream/lard base.

4. richness:
   - 1.0: Very light, water-based broth or fresh greens.
   - 5.0: Hearty but balanced (e.g., standard pasta or chicken stew).
   - 10.0: Decadent, extremely heavy, dense, and highly caloric.

5. ingredient_diversity:
   - 1.0: 1 to 3 basic ingredients.
   - 5.0: 5 to 8 ingredients including some basic spices.
   - 10.0: 15+ ingredients crossing multiple flavor profiles and categories.

6. complexity_of_preparation:
   - 1.0: Mix and eat, zero active cooking required.
   - 5.0: Standard 30-minute active cooking, basic chopping and heating.
   - 10.0: Multi-day prep, requiring advanced techniques (e.g., sous-vide, fermentation, intricate folding).

Return ONLY a strictly valid JSON array containing exactly {len(recipes)} objects. Do not include markdown formatting like ```json.
Each object MUST adhere to this structure:
{{
  "recipe_id": number,
  "presentation_quality": float,
  "saltiness": float,
  "oiliness": float,
  "richness": float,
  "ingredient_diversity": float,
  "complexity_of_preparation": float
}}

Input JSON:
{recipes_json}
"""


@retry(
    retry=retry_if_exception_type(LlmResponseError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def score_recipes(client: genai.Client, recipes: List[Recipe]) -> Dict[int, Dict[str, float]]:
    prompt = build_scoring_prompt(recipes)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    if not response or not response.text:
        raise LlmResponseError("Empty response from model")

    parsed = _extract_json_array(response.text)
    if len(parsed) != len(recipes):
        raise LlmResponseError("Unexpected JSON array length for scores")

    scores: Dict[int, Dict[str, float]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise LlmResponseError("Non-object entry in scoring output")
        if "recipe_id" not in item:
            raise LlmResponseError("Missing recipe_id in scoring output")
        for key in FEATURE_COLUMNS:
            if key not in item:
                raise LlmResponseError(f"Missing key in scoring output: {key}")
        scores[int(item["recipe_id"])] = {
            key: float(item[key]) for key in FEATURE_COLUMNS
        }
    return scores


def ingredient_count(text: str) -> int:
    return len([item for item in text.split(",") if item.strip()])


def steps_length(text: str) -> int:
    return len(text)


def build_baseline_features_from_df(data: pd.DataFrame) -> np.ndarray:
    ingredients = data["ingredients"].fillna("")
    steps = data["steps"].fillna("")
    counts = ingredients.apply(ingredient_count)
    lengths = steps.apply(steps_length)
    return np.vstack([counts.to_numpy(), lengths.to_numpy()]).T


def build_baseline_features_from_recipe(recipe: Recipe) -> np.ndarray:
    return np.array([[ingredient_count(recipe.ingredients), steps_length(recipe.steps)]])


def load_or_train_combined_model(
    augmented_path: str,
    interactions_path: str,
    min_reviews: int,
    model_path: str,
    scaler_path: str,
) -> tuple[StandardScaler, LogisticRegression]:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler, model

    data = pd.read_csv(augmented_path)
    ratings = load_avg_ratings(interactions_path, min_reviews)
    data = data.merge(ratings, on="recipe_id", how="inner")
    missing = [
        col
        for col in FEATURE_COLUMNS + ["avg_rating", "ingredients", "steps"]
        if col not in data.columns
    ]
    if missing:
        raise RuntimeError(f"Missing required columns in augmented data: {missing}")

    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating", "ingredients", "steps"]).copy()
    median_rating = data["avg_rating"].median()
    data["target"] = (data["avg_rating"] > median_rating).astype(int)

    llm_features = data[FEATURE_COLUMNS].to_numpy()
    baseline_features = build_baseline_features_from_df(data)
    combined_features = np.hstack([llm_features, baseline_features])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_features)

    model = LogisticRegression(max_iter=400)
    model.fit(combined_scaled, data["target"].to_numpy())

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    return scaler, model


def load_avg_ratings(interactions_path: str, min_reviews: int) -> pd.DataFrame:
    interactions = load_and_prepare_interactions(interactions_path)
    grouped = (
        interactions.groupby("recipe_id", as_index=False)["rating"]
        .agg(avg_rating="mean", rating_count="count")
    )
    grouped = grouped[grouped["rating_count"] >= min_reviews].copy()
    grouped.drop(columns=["rating_count"], inplace=True)
    grouped["recipe_id"] = pd.to_numeric(grouped["recipe_id"], errors="coerce").astype(
        "Int64"
    )
    return grouped


def format_recipe_block(title: str, recipe: Recipe) -> str:
    return (
        f"{title}\n"
        f"Name: {recipe.name}\n"
        f"Ingredients: {recipe.ingredients}\n"
        f"Steps: {recipe.steps}\n"
    )


async def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from environment")

    augmented_path = get_output_path(PROCESSED_DIR)
    interactions_path = get_interactions_path(RAW_DIR)
    if not os.path.exists(augmented_path):
        raise RuntimeError(
            "Augmented dataset not found. Run `uv run cs568-prepare` first."
        )

    client = genai.Client(api_key=api_key)

    optimized_recipe = await asyncio.to_thread(optimize_recipe, client, BASELINE_RECIPE)
    scores = await asyncio.to_thread(
        score_recipes, client, [BASELINE_RECIPE, optimized_recipe]
    )

    scaler, model = load_or_train_combined_model(
        augmented_path, interactions_path, MIN_REVIEWS, MODEL_PATH, SCALER_PATH
    )

    baseline_llm = np.array([scores[BASELINE_RECIPE.recipe_id][key] for key in FEATURE_COLUMNS])
    optimized_llm = np.array([scores[optimized_recipe.recipe_id][key] for key in FEATURE_COLUMNS])

    baseline_baseline = build_baseline_features_from_recipe(BASELINE_RECIPE)
    optimized_baseline = build_baseline_features_from_recipe(optimized_recipe)

    baseline_combined = np.hstack([baseline_llm, baseline_baseline.flatten()]).reshape(1, -1)
    optimized_combined = np.hstack([optimized_llm, optimized_baseline.flatten()]).reshape(1, -1)

    baseline_scaled = scaler.transform(baseline_combined)
    optimized_scaled = scaler.transform(optimized_combined)

    baseline_prob = model.predict_proba(baseline_scaled)[0, 1]
    optimized_prob = model.predict_proba(optimized_scaled)[0, 1]

    divider = "=" * 41
    print(divider)
    print(format_recipe_block("[1] THE BASELINE RECIPE", BASELINE_RECIPE))
    print(format_recipe_block(
        "[2] THE OPTIMIZED RECIPE (Air Fryer/Pressure Cooker Modernization)",
        optimized_recipe,
    ))
    print("[3] FEATURE SHIFT ANALYSIS")
    for feature in FEATURE_COLUMNS:
        before = scores[BASELINE_RECIPE.recipe_id][feature]
        after = scores[optimized_recipe.recipe_id][feature]
        print(f"- {feature}: {before:.1f} -> {after:.1f}")
    print("[4] PREDICTION SHOWDOWN")
    print(f"- Baseline Win Probability: {baseline_prob * 100:.1f}%")
    print(f"- Optimized Win Probability: {optimized_prob * 100:.1f}%")
    print(divider)


if __name__ == "__main__":
    asyncio.run(main())
