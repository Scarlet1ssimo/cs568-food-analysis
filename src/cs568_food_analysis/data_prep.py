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
    user_id_col = _find_column(interactions.columns, ["user_id", "user", "author_id"])
    recipe_id_col = _find_column(interactions.columns, ["recipe_id", "id"])
    rating_col = _find_column(interactions.columns, ["rating", "stars"])

    user_stats = (
        interactions[[user_id_col, rating_col]]
        .groupby(user_id_col)[rating_col]
        .agg(["std", "count"])
        .reset_index()
    )
    unreliable_users = user_stats[
        (user_stats["count"] > 1) & (user_stats["std"] == 0.0)
    ][user_id_col]
    filtered = interactions[~interactions[user_id_col].isin(unreliable_users)]

    filtered = filtered[[recipe_id_col, rating_col]].copy()
    filtered.rename(
        columns={recipe_id_col: "recipe_id", rating_col: "rating"},
        inplace=True,
    )
    filtered["recipe_id"] = pd.to_numeric(filtered["recipe_id"], errors="coerce").astype(
        "Int64"
    )
    filtered["rating"] = pd.to_numeric(filtered["rating"], errors="coerce")
    return filtered


def filter_recipes_by_review_count(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    min_reviews: int,
) -> pd.DataFrame:
    counts = (
        interactions.groupby("recipe_id", as_index=False)
        .size()
        .rename(columns={"size": "rating_count"})
    )
    valid_ids = counts[counts["rating_count"] >= min_reviews]["recipe_id"]
    return recipes[recipes["recipe_id"].isin(valid_ids)].copy()


def sample_recipes(merged: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    return merged.sample(n=sample_size, random_state=42).reset_index(drop=True)
