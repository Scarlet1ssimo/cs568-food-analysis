import asyncio
import ast
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "augmented_recipes.csv")

RECIPES_PATH = os.path.join(RAW_DIR, "RAW_recipes.csv")
INTERACTIONS_PATH = os.path.join(RAW_DIR, "RAW_interactions.csv")

MODEL_NAME = "gemma-3-27b-it"
BATCH_SIZE = 10
SAMPLE_SIZE = 20
SLEEP_SECONDS = 12

FEATURE_COLUMNS = [
    "presentation_quality",
    "saltiness",
    "oiliness",
    "richness",
    "ingredient_diversity",
    "complexity_of_preparation",
]


class LlmResponseError(RuntimeError):
    pass


@dataclass
class RecipeRecord:
    recipe_id: int
    name: str
    ingredients: str
    steps: str


def _safe_literal_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            return [value]
        return [value]
    return [str(value)]


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Missing required column; tried {list(candidates)}")


def load_and_prepare_recipes(recipes_path: str) -> pd.DataFrame:
    recipes = pd.read_csv(recipes_path)
    id_col = _find_column(recipes.columns, ["id", "recipe_id"])
    name_col = _find_column(recipes.columns, ["name", "recipe_name"])
    ingredients_col = _find_column(recipes.columns, ["ingredients"])
    steps_col = _find_column(recipes.columns, ["steps"])

    recipes = recipes[[id_col, name_col, ingredients_col, steps_col]].copy()
    recipes.rename(
        columns={
            id_col: "recipe_id",
            name_col: "name",
            ingredients_col: "ingredients",
            steps_col: "steps",
        },
        inplace=True,
    )
    recipes["recipe_id"] = pd.to_numeric(recipes["recipe_id"], errors="coerce").astype("Int64")

    recipes["ingredients"] = recipes["ingredients"].apply(
        lambda value: ", ".join(_safe_literal_list(value))
    )
    recipes["steps"] = recipes["steps"].apply(
        lambda value: " ".join(_safe_literal_list(value))
    )
    return recipes


def load_and_prepare_interactions(interactions_path: str) -> pd.DataFrame:
    interactions = pd.read_csv(interactions_path)
    recipe_id_col = _find_column(interactions.columns, ["recipe_id", "id"])
    rating_col = _find_column(interactions.columns, ["rating", "stars"])
    grouped = (
        interactions[[recipe_id_col, rating_col]]
        .groupby(recipe_id_col, as_index=False)
        .mean()
    )
    grouped.rename(
        columns={recipe_id_col: "recipe_id", rating_col: "avg_rating"},
        inplace=True,
    )
    grouped["recipe_id"] = pd.to_numeric(grouped["recipe_id"], errors="coerce").astype("Int64")
    return grouped


def merge_recipes_and_ratings(recipes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    merged = recipes.merge(ratings, on="recipe_id", how="inner")
    return merged.dropna(subset=["avg_rating"])


def sample_recipes(merged: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    return merged.sample(n=sample_size, random_state=42).reset_index(drop=True)


def chunk_dataframe(frame: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    return [frame.iloc[start: start + batch_size] for start in range(0, len(frame), batch_size)]

def build_prompt(batch: pd.DataFrame) -> str:
    payload = [
        RecipeRecord(
            recipe_id=int(row.recipe_id),
            name=str(row.name),
            ingredients=str(row.ingredients),
            steps=str(row.steps),
        )
        for row in batch.itertuples(index=False)
    ]
    batch_size = len(payload)
    recipes_json = json.dumps(
        [record.__dict__ for record in payload], ensure_ascii=True)
    return f"""You are an expert culinary data scientist extracting quantitative features from recipe texts.
Analyze the following {batch_size} recipes and assign continuous scores (from 1.0 to 10.0, allowing one decimal place) for six specific attributes.

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

Return ONLY a strictly valid JSON array containing exactly {batch_size} objects. Do not include markdown formatting like ```json.
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
def generate_batch_features(client: genai.Client, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt)
    if not response or not response.text:
        raise LlmResponseError("Empty response from model")
    try:
        start = response.text.find("[")
        end = response.text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("Missing JSON array brackets")
        parsed = json.loads(response.text[start: end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        raise LlmResponseError("Failed to parse JSON response") from exc

    if not isinstance(parsed, list) or len(parsed) != batch_size:
        raise LlmResponseError("Unexpected JSON array length")
    for item in parsed:
        if not isinstance(item, dict):
            raise LlmResponseError("Non-object entry in JSON array")
        missing = [key for key in ["recipe_id",
                                   *FEATURE_COLUMNS] if key not in item]
        if missing:
            raise LlmResponseError(f"Missing keys in LLM output: {missing}")
    return parsed


async def augment_with_llm(sampled: pd.DataFrame) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from environment")

    client = genai.Client(api_key=api_key)
    batches = chunk_dataframe(sampled, BATCH_SIZE)
    outputs: List[Dict[str, Any]] = []

    for index, batch in enumerate(batches):
        prompt = build_prompt(batch)
        batch_features = await asyncio.to_thread(
            generate_batch_features, client, prompt, len(batch)
        )
        outputs.extend(batch_features)
        if index < len(batches) - 1:
            await asyncio.sleep(SLEEP_SECONDS)

    features = pd.DataFrame(outputs)
    features["recipe_id"] = pd.to_numeric(features["recipe_id"], errors="coerce").astype("Int64")
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return sampled.merge(features, on="recipe_id", how="inner")


def run_logistic_regression(augmented_path: str) -> None:
    data = pd.read_csv(augmented_path)
    missing = [col for col in FEATURE_COLUMNS +
               ["avg_rating"] if col not in data.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in augmented data: {missing}")

    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating"]).copy()
    data["target"] = (data["avg_rating"] >= 4.5).astype(int)
    X = data[FEATURE_COLUMNS].to_numpy()
    y = data["target"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)

    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient": model.coef_[0]}
    )
    coefficients["abs_weight"] = coefficients["coefficient"].abs()
    coefficients.sort_values("abs_weight", ascending=False, inplace=True)

    logging.info("Feature importances (sorted by absolute weight):")
    print(coefficients[["feature", "coefficient"]].to_string(index=False))


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    recipes = load_and_prepare_recipes(RECIPES_PATH)
    ratings = load_and_prepare_interactions(INTERACTIONS_PATH)
    merged = merge_recipes_and_ratings(recipes, ratings)
    sampled = sample_recipes(merged, SAMPLE_SIZE)

    logging.info("Augmenting %s recipes with LLM features...", len(sampled))
    augmented = await augment_with_llm(sampled)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    augmented.to_csv(OUTPUT_PATH, index=False)
    logging.info("Saved augmented data to %s", OUTPUT_PATH)

    run_logistic_regression(OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
