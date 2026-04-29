"""Configuration defaults and helpers."""

import os
from typing import List

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILENAME = "augmented_recipes.csv"

MODEL_NAME = "gemma-3-27b-it"
BATCH_SIZE = 10
SAMPLE_SIZE = 1000
SLEEP_SECONDS = 12
TARGET_RATING = 4.5
MAX_ITER = 200

FEATURE_COLUMNS: List[str] = [
    "presentation_quality",
    "saltiness",
    "oiliness",
    "richness",
    "ingredient_diversity",
    "complexity_of_preparation",
]


def get_recipes_path(raw_dir: str) -> str:
    return os.path.join(raw_dir, "RAW_recipes.csv")


def get_interactions_path(raw_dir: str) -> str:
    return os.path.join(raw_dir, "RAW_interactions.csv")


def get_output_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, OUTPUT_FILENAME)
