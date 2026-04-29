"""Model training and analysis utilities."""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


def load_augmented_data(augmented_path: str) -> pd.DataFrame:
    return pd.read_csv(augmented_path)


def median_split_targets(data: pd.DataFrame) -> pd.DataFrame:
    median_rating = data["avg_rating"].median()
    split = data.copy()
    split["target"] = (split["avg_rating"] > median_rating).astype(int)
    return split


def run_logistic_regression_from_df(
    data: pd.DataFrame,
    target_rating: float,
    max_iter: int,
) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS + ["avg_rating"] if col not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in augmented data: {missing}")

    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating"]).copy()
    split = median_split_targets(data)
    X = split[FEATURE_COLUMNS].to_numpy()
    y = split["target"].to_numpy()

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


def run_logistic_regression(
    augmented_path: str,
    target_rating: float,
    max_iter: int,
) -> pd.DataFrame:
    data = load_augmented_data(augmented_path)
    return run_logistic_regression_from_df(data, target_rating, max_iter)


def compute_top_bottom_means(
    data: pd.DataFrame,
    top_quantile: float = 0.9,
    bottom_quantile: float = 0.1,
) -> Tuple[pd.Series, pd.Series]:
    top_threshold = data["avg_rating"].quantile(top_quantile)
    bottom_threshold = data["avg_rating"].quantile(bottom_quantile)

    top_group = data[data["avg_rating"] >= top_threshold]
    bottom_group = data[data["avg_rating"] <= bottom_threshold]

    top_means = top_group[FEATURE_COLUMNS].mean()
    bottom_means = bottom_group[FEATURE_COLUMNS].mean()
    return top_means, bottom_means


def plot_radar_comparison(
    top_means: pd.Series,
    bottom_means: pd.Series,
    title: str = "Top 10% vs Bottom 10% Feature Profiles",
    output_path: str | None = None,
) -> None:
    labels = FEATURE_COLUMNS
    values_top = top_means[labels].to_numpy()
    values_bottom = bottom_means[labels].to_numpy()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    values_top = np.concatenate([values_top, values_top[:1]])
    values_bottom = np.concatenate([values_bottom, values_bottom[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, values_top, linewidth=2, label="Top 10%")
    ax.fill(angles, values_top, alpha=0.2)
    ax.plot(angles, values_bottom, linewidth=2, label="Bottom 10%")
    ax.fill(angles, values_bottom, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    ax.set_ylim(1.0, 10.0)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def build_baseline_features(data: pd.DataFrame) -> pd.DataFrame:
    if "ingredients" not in data.columns or "steps" not in data.columns:
        raise RuntimeError("Missing ingredients or steps columns for baseline features")

    ingredients = data["ingredients"].fillna("")
    ingredient_count = ingredients.apply(
        lambda value: len([item for item in value.split(",") if item.strip()])
    )

    return pd.DataFrame({"ingredient_count": ingredient_count})


def plot_roc_curves(
    y_test: np.ndarray,
    llm_scores: np.ndarray,
    baseline_scores: np.ndarray,
    combined_scores: np.ndarray,
    output_path: str | None = None,
) -> None:
    llm_fpr, llm_tpr, _ = roc_curve(y_test, llm_scores)
    baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_scores)
    combined_fpr, combined_tpr, _ = roc_curve(y_test, combined_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(llm_fpr, llm_tpr, label="LLM Features")
    ax.plot(baseline_fpr, baseline_tpr, label="Baseline")
    ax.plot(combined_fpr, combined_tpr, label="Combined")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", color="black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bad", "Good"])
    ax.set_yticklabels(["Bad", "Good"])

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def evaluate_models(
    data: pd.DataFrame,
    target_rating: float,
    max_iter: int,
    output_dir: str | None = None,
) -> None:
    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating"]).copy()
    split = median_split_targets(data)
    y = split["target"].to_numpy()

    llm_features = split[FEATURE_COLUMNS].to_numpy()
    baseline_features = build_baseline_features(split).to_numpy()
    combined_features = np.hstack([llm_features, baseline_features])

    (
        llm_train,
        llm_test,
        baseline_train,
        baseline_test,
        combined_train,
        combined_test,
        y_train,
        y_test,
    ) = train_test_split(
        llm_features,
        baseline_features,
        combined_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    llm_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
    )
    baseline_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
    )
    combined_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
    )

    llm_model.fit(llm_train, y_train)
    baseline_model.fit(baseline_train, y_train)
    combined_model.fit(combined_train, y_train)

    llm_scores = llm_model.predict_proba(llm_test)[:, 1]
    baseline_scores = baseline_model.predict_proba(baseline_test)[:, 1]
    combined_scores = combined_model.predict_proba(combined_test)[:, 1]

    llm_pred = (llm_scores >= 0.5).astype(int)
    baseline_pred = (baseline_scores >= 0.5).astype(int)
    combined_pred = (combined_scores >= 0.5).astype(int)

    llm_accuracy = accuracy_score(y_test, llm_pred)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    combined_accuracy = accuracy_score(y_test, combined_pred)
    llm_auc = roc_auc_score(y_test, llm_scores)
    baseline_auc = roc_auc_score(y_test, baseline_scores)
    combined_auc = roc_auc_score(y_test, combined_scores)

    logging.info(
        "LLM features: accuracy=%.3f, ROC-AUC=%.3f", llm_accuracy, llm_auc
    )
    logging.info(
        "Baseline features: accuracy=%.3f, ROC-AUC=%.3f",
        baseline_accuracy,
        baseline_auc,
    )
    logging.info(
        "Combined features: accuracy=%.3f, ROC-AUC=%.3f",
        combined_accuracy,
        combined_auc,
    )

    roc_path = None
    cm_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        roc_path = os.path.join(output_dir, "roc_curve.png")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")

    plot_roc_curves(
        y_test,
        llm_scores,
        baseline_scores,
        combined_scores,
        output_path=roc_path,
    )
    plot_confusion_matrix(confusion_matrix(y_test, llm_pred), output_path=cm_path)


def analyze_with_radar_plot(
    augmented_path: str,
    target_rating: float,
    max_iter: int,
    output_path: str | None = None,
    eval_output_dir: str | None = None,
) -> None:
    data = load_augmented_data(augmented_path)
    run_logistic_regression_from_df(data, target_rating, max_iter)
    data = data.dropna(subset=FEATURE_COLUMNS + ["avg_rating"]).copy()
    top_means, bottom_means = compute_top_bottom_means(data)
    plot_radar_comparison(top_means, bottom_means, output_path=output_path)
    evaluate_models(data, target_rating, max_iter, output_dir=eval_output_dir)
