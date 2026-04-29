"""Legacy entrypoint that runs both pipeline stages."""

import logging

from cs568_food_analysis.cli import analyze_data, prepare_data
from cs568_food_analysis.config import (
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
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    output_path = prepare_data(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        sample_size=SAMPLE_SIZE,
        batch_size=BATCH_SIZE,
        sleep_seconds=SLEEP_SECONDS,
        model_name=MODEL_NAME,
    )
    analyze_data(
        augmented_path=output_path or get_output_path(PROCESSED_DIR),
        interactions_path=get_interactions_path(RAW_DIR),
        target_rating=TARGET_RATING,
        max_iter=MAX_ITER,
    )


if __name__ == "__main__":
    main()
