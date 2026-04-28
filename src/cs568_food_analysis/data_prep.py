"""Data loading and preparation utilities."""

import ast
from typing import Any, Iterable, List

import pandas as pd


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
    recipes["recipe_id"] = pd.to_numeric(recipes["recipe_id"], errors="coerce").astype(
        "Int64"
    )

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
    grouped["recipe_id"] = pd.to_numeric(grouped["recipe_id"], errors="coerce").astype(
        "Int64"
    )
    return grouped


def merge_recipes_and_ratings(recipes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    merged = recipes.merge(ratings, on="recipe_id", how="inner")
    return merged.dropna(subset=["avg_rating"])


def sample_recipes(merged: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    return merged.sample(n=sample_size, random_state=42).reset_index(drop=True)
