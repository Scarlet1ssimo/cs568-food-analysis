"""LLM-based recipe augmentation."""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import FEATURE_COLUMNS


class LlmResponseError(RuntimeError):
    pass


@dataclass
class RecipeRecord:
    recipe_id: int
    name: str
    ingredients: str
    steps: str


def chunk_dataframe(frame: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    return [
        frame.iloc[start : start + batch_size]
        for start in range(0, len(frame), batch_size)
    ]


def build_prompt(batch: pd.DataFrame) -> str:
    payload = [
        RecipeRecord(
            recipe_id=int(str(row.recipe_id)),
            name=str(row.name),
            ingredients=str(row.ingredients),
            steps=str(row.steps),
        )
        for row in batch.itertuples(index=False)
    ]
    batch_size = len(payload)
    recipes_json = json.dumps([record.__dict__ for record in payload], ensure_ascii=True)
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


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: List[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict) and "text" in item:
                fragments.append(str(item["text"]))
        return "\n".join(fragments)
    return ""


def create_llm_client(llm_provider: str, model_name: str) -> Any:
    provider = llm_provider.strip().lower()
    if provider in {"google", "gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from environment")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from environment")
        return ChatOpenAI(model=model_name, temperature=0)

    raise ValueError(
        f"Unsupported llm provider '{llm_provider}'. Supported values: google, gemini, openai"
    )


@retry(
    retry=retry_if_exception_type(LlmResponseError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def generate_batch_features(
    client: Any,
    prompt: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    response = client.invoke(prompt)
    response_text = _extract_response_text(response.content)
    if not response_text:
        raise LlmResponseError("Empty response from model")
    try:
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("Missing JSON array brackets")
        parsed = json.loads(response_text[start : end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        raise LlmResponseError("Failed to parse JSON response") from exc

    if not isinstance(parsed, list) or len(parsed) != batch_size:
        raise LlmResponseError("Unexpected JSON array length")
    for item in parsed:
        if not isinstance(item, dict):
            raise LlmResponseError("Non-object entry in JSON array")
        missing = [key for key in ["recipe_id", *FEATURE_COLUMNS] if key not in item]
        if missing:
            raise LlmResponseError(f"Missing keys in LLM output: {missing}")
    return parsed


async def augment_with_llm(
    sampled: pd.DataFrame,
    batch_size: int,
    sleep_seconds: int,
    llm_provider: str,
    model_name: str,
) -> pd.DataFrame:
    load_dotenv()
    client = create_llm_client(llm_provider, model_name)
    batches = chunk_dataframe(sampled, batch_size)
    outputs: List[Dict[str, Any]] = []

    for index, batch in enumerate(batches):
        prompt = build_prompt(batch)
        batch_features = await asyncio.to_thread(
            generate_batch_features,
            client,
            prompt,
            len(batch),
        )
        outputs.extend(batch_features)
        if index < len(batches) - 1:
            await asyncio.sleep(sleep_seconds)

    features = pd.DataFrame(outputs)
    features["recipe_id"] = pd.to_numeric(features["recipe_id"], errors="coerce").astype(
        "Int64"
    )
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return sampled.merge(features, on="recipe_id", how="inner")
