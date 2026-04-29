"""Command-line entrypoints for the two-stage pipeline."""

import argparse
import asyncio
import logging
import os

from .analysis import analyze_with_radar_plot
from .config import (
    BATCH_SIZE,
    MIN_REVIEWS,
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
    filter_recipes_by_review_count,
    load_and_prepare_interactions,
    load_and_prepare_recipes,
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
    interactions = load_and_prepare_interactions(interactions_path)
    eligible = filter_recipes_by_review_count(recipes, interactions, MIN_REVIEWS)
    sampled = sample_recipes(eligible, sample_size)

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
    interactions_path: str,
    target_rating: float,
    max_iter: int,
    plot_path: str | None = None,
    eval_output_dir: str | None = None,
) -> None:
    analyze_with_radar_plot(
        augmented_path,
        interactions_path,
        target_rating,
        max_iter,
        plot_path,
        eval_output_dir,
    )


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
    parser.add_argument(
        "--interactions-path",
        default=get_interactions_path(RAW_DIR),
        help="Path to raw interactions CSV for avg rating calculation.",
    )
    parser.add_argument("--target-rating", type=float, default=TARGET_RATING)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional file path to save the radar plot instead of showing it.",
    )
    parser.add_argument(
        "--eval-output-dir",
        default=PROCESSED_DIR,
        help="Directory to save ROC and confusion matrix plots.",
    )
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
        interactions_path=args.interactions_path,
        target_rating=args.target_rating,
        max_iter=args.max_iter,
        plot_path=args.plot_path,
        eval_output_dir=args.eval_output_dir,
    )
