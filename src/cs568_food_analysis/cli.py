"""Command-line entrypoints for the two-stage pipeline."""

import argparse
import asyncio
import logging
import os

from .analysis import run_logistic_regression
from .config import (
    BATCH_SIZE,
    MAX_ITER,
    MODEL_NAME,
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLE_SIZE,
    SLEEP_SECONDS,
    TARGET_RATING,
    get_interactions_path,
    get_output_path,
    get_recipes_path,
)
from .data_prep import (
    load_and_prepare_interactions,
    load_and_prepare_recipes,
    merge_recipes_and_ratings,
    sample_recipes,
)
from .llm_augment import augment_with_llm


def prepare_data(
    raw_dir: str,
    processed_dir: str,
    sample_size: int,
    batch_size: int,
    sleep_seconds: int,
    model_name: str,
    output_path: str | None = None,
) -> str:
    recipes_path = get_recipes_path(raw_dir)
    interactions_path = get_interactions_path(raw_dir)
    resolved_output = output_path or get_output_path(processed_dir)

    recipes = load_and_prepare_recipes(recipes_path)
    ratings = load_and_prepare_interactions(interactions_path)
    merged = merge_recipes_and_ratings(recipes, ratings)
    sampled = sample_recipes(merged, sample_size)

    logging.info("Augmenting %s recipes with LLM features...", len(sampled))
    augmented = asyncio.run(
        augment_with_llm(sampled, batch_size, sleep_seconds, model_name)
    )
    os.makedirs(processed_dir, exist_ok=True)
    augmented.to_csv(resolved_output, index=False)
    logging.info("Saved augmented data to %s", resolved_output)
    return resolved_output


def analyze_data(
    augmented_path: str,
    target_rating: float,
    max_iter: int,
) -> None:
    run_logistic_regression(augmented_path, target_rating, max_iter)


def _build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: prepare data and augment recipes with LLM features.",
    )
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sleep-seconds", type=int, default=SLEEP_SECONDS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-path", default=None)
    return parser


def _build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: analyze processed data with logistic regression.",
    )
    parser.add_argument(
        "--augmented-path",
        default=get_output_path(PROCESSED_DIR),
    )
    parser.add_argument("--target-rating", type=float, default=TARGET_RATING)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    return parser


def main_prepare() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_prepare_parser()
    args = parser.parse_args()
    prepare_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        model_name=args.model_name,
        output_path=args.output_path,
    )


def main_analyze() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_analyze_parser()
    args = parser.parse_args()
    analyze_data(
        augmented_path=args.augmented_path,
        target_rating=args.target_rating,
        max_iter=args.max_iter,
    )
