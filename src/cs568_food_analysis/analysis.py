"""Model training and analysis utilities."""

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


def run_logistic_regression(
    augmented_path: str,
    target_rating: float,
    max_iter: int,
) -> pd.DataFrame:
    data = pd.read_csv(augmented_path)
    missing = [col for col in FEATURE_COLUMNS + ["avg_rating"] if col not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in augmented data: {missing}")

    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating"]).copy()
    data["target"] = (data["avg_rating"] >= target_rating).astype(int)
    X = data[FEATURE_COLUMNS].to_numpy()
    y = data["target"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_scaled, y)

    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient": model.coef_[0]}
    )
    coefficients["abs_weight"] = coefficients["coefficient"].abs()
    coefficients.sort_values("abs_weight", ascending=False, inplace=True)

    logging.info("Feature importances (sorted by absolute weight):")
    print(coefficients[["feature", "coefficient"]].to_string(index=False))
    return coefficients
