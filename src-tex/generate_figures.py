"""Reproduce the report figures and numerical tables from the project data.

Run from the repository root with:
    uv run --group dev python src-tex/generate_figures.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_DIR / "figures"
GENERATED_DIR = REPORT_DIR / "generated"
DATA_PATH = ROOT / "dataset" / "pima_indians_diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DISPLAY_NAMES = OrderedDict(
    [
        ("Custom KNN", "Custom KNN"),
        ("Decision Tree", "Decision Tree"),
        ("SVM", "Calibrated SVM"),
        ("Gaussian NB", "Gaussian NB"),
        ("MLP", "MLP"),
    ]
)
COLORS = ["#2F5597", "#C55A11", "#548235", "#8064A2", "#4BACC6"]


class KNNFromScratch(ClassifierMixin, BaseEstimator):
    """Small Euclidean KNN implementation matching the training notebook."""

    def __init__(self, k: int = 7) -> None:
        self.k = k

    def fit(self, features: np.ndarray, labels: pd.Series) -> "KNNFromScratch":
        self.training_features_ = np.asarray(features)
        self.training_labels_ = np.asarray(labels, dtype=int)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        test_features = np.asarray(features)
        distances = np.sqrt(
            np.sum((test_features[:, np.newaxis, :] - self.training_features_[np.newaxis, :, :]) ** 2, axis=2)
        )
        neighbor_indices = np.argsort(distances, axis=1)[:, : self.k]
        positive_probability = self.training_labels_[neighbor_indices].mean(axis=1)
        return np.column_stack((1 - positive_probability, positive_probability))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] > 0.5).astype(int)


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_preprocessor() -> Pipeline:
    """Match the notebook's training-only preprocessing exactly."""

    return Pipeline(
        [
            (
                "zero_imputer",
                ColumnTransformer(
                    transformers=[
                        (
                            "invalid_zero_imputer",
                            SimpleImputer(missing_values=0, strategy="median"),
                            ZERO_AS_MISSING,
                        )
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def build_models() -> OrderedDict[str, object]:
    svm = CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        method="sigmoid",
        cv=5,
        ensemble=False,
    )
    return OrderedDict(
        [
            ("Custom KNN", KNNFromScratch(k=7)),
            (
                "Decision Tree",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=4,
                    min_samples_split=10,
                    random_state=42,
                ),
            ),
            ("SVM", svm),
            ("Gaussian NB", GaussianNB()),
            (
                "MLP",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y_true: pd.Series, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    labels = [
        "Load and audit\nthe dataset",
        "Create a stratified\n80/20 split",
        "Fit preprocessing\non training data",
        "Train five\nclassifiers",
        "Evaluate once on the\nheld-out test set",
        "Save the MLP pipeline\nfor the demo app",
    ]
    positions = [(0.17, 0.72), (0.50, 0.72), (0.83, 0.72), (0.83, 0.25), (0.50, 0.25), (0.17, 0.25)]
    box_width, box_height = 0.25, 0.26

    for index, ((x_pos, y_pos), label) in enumerate(zip(positions, labels, strict=True)):
        box = FancyBboxPatch(
            (x_pos - box_width / 2, y_pos - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#EAF0F8" if index % 2 == 0 else "#FDF0E7",
            edgecolor=COLORS[index % len(COLORS)],
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x_pos, y_pos, label, ha="center", va="center", fontsize=8.2)
        if index < len(labels) - 1:
            next_x, next_y = positions[index + 1]
            if np.isclose(x_pos, next_x):
                start = (x_pos, y_pos - box_height / 2 - 0.012)
                end = (next_x, next_y + box_height / 2 + 0.012)
            elif next_x > x_pos:
                start = (x_pos + box_width / 2 + 0.012, y_pos)
                end = (next_x - box_width / 2 - 0.012, next_y)
            else:
                start = (x_pos - box_width / 2 - 0.012, y_pos)
                end = (next_x + box_width / 2 + 0.012, next_y)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
            )

    fig.savefig(FIGURE_DIR / "workflow.pdf")
    plt.close(fig)


def draw_dataset_audit(df: pd.DataFrame) -> None:
    class_counts = df["Outcome"].value_counts().sort_index()
    zero_counts = (df[ZERO_AS_MISSING] == 0).sum().sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    bars = axes[0].bar(
        ["No diabetes (0)", "Diabetes (1)"],
        class_counts.values,
        color=[COLORS[0], COLORS[1]],
        width=0.62,
    )
    axes[0].set_title("Outcome distribution")
    axes[0].set_ylabel("Records")
    axes[0].set_ylim(0, 560)
    axes[0].bar_label(bars, padding=3)
    axes[0].grid(axis="x", visible=False)

    bars = axes[1].barh(zero_counts.index, zero_counts.values, color=COLORS[2])
    axes[1].set_title("Zeros treated as missing")
    axes[1].set_xlabel("Zero-valued records")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xlim(0, 410)
    axes[1].grid(axis="y", visible=False)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIGURE_DIR / "dataset_audit.pdf")
    plt.close(fig)


def draw_correlation_heatmap(df: pd.DataFrame) -> None:
    analysis_df = df.copy()
    analysis_df[ZERO_AS_MISSING] = analysis_df[ZERO_AS_MISSING].replace(0, np.nan)
    correlations = analysis_df.corr(numeric_only=True)
    short_names = {
        "Pregnancies": "Preg.",
        "Glucose": "Glucose",
        "BloodPressure": "BP",
        "SkinThickness": "Skin",
        "Insulin": "Insulin",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "DPF",
        "Age": "Age",
        "Outcome": "Outcome",
    }
    correlations = correlations.rename(index=short_names, columns=short_names)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    sns.heatmap(
        correlations,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5},
        cbar_kws={"label": "Pearson correlation", "shrink": 0.72},
        ax=ax,
    )
    ax.set_title("Pairwise correlations after treating selected zeros as missing")
    ax.tick_params(axis="x", rotation=42)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.pdf")
    plt.close(fig)


def draw_confusion_matrices(results: OrderedDict[str, dict[str, float | int]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6))
    for ax, (name, values) in zip(axes.flat, results.items(), strict=False):
        matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.6,
            linecolor="white",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            ax=ax,
        )
        ax.set_title(DISPLAY_NAMES[name], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="both", rotation=0)
    axes.flat[-1].axis("off")
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    fig.savefig(FIGURE_DIR / "confusion_matrices.pdf")
    plt.close(fig)


def draw_model_comparison(results: OrderedDict[str, dict[str, float | int]]) -> None:
    metric_keys = ["accuracy", "sensitivity", "specificity", "roc_auc", "mcc"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC", "MCC"]
    names = list(results)
    x_positions = np.arange(len(metric_keys))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    for index, name in enumerate(names):
        values = [float(results[name][key]) for key in metric_keys]
        ax.bar(
            x_positions + (index - 2) * bar_width,
            values,
            width=bar_width,
            label=DISPLAY_NAMES[name],
            color=COLORS[index],
        )
    ax.set_xticks(x_positions, metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test performance")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.pdf")
    plt.close(fig)


def draw_roc_curves(y_test: pd.Series, scores: OrderedDict[str, np.ndarray], results: OrderedDict[str, dict[str, float | int]]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for index, (name, model_scores) in enumerate(scores.items()):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, model_scores)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            lw=1.7,
            color=COLORS[index],
            label=f"{DISPLAY_NAMES[name]} ({float(results[name]['roc_auc']):.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1, label="Chance (0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", frameon=True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curves.pdf")
    plt.close(fig)


def draw_permutation_importance(
    preprocessor: Pipeline,
    mlp_model: MLPClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    fitted_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", mlp_model)]
    )
    importance = permutation_importance(
        fitted_pipeline,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
        n_jobs=1,
    )
    order = np.argsort(importance.importances_mean)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.barh(
        np.asarray(FEATURES)[order],
        importance.importances_mean[order],
        xerr=importance.importances_std[order],
        color=COLORS[0],
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Decrease in test ROC-AUC after permutation")
    ax.set_title("Exploratory permutation importance for the MLP")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance.pdf")
    plt.close(fig)

    return {
        feature: {
            "mean": float(importance.importances_mean[index]),
            "std": float(importance.importances_std[index]),
        }
        for index, feature in enumerate(FEATURES)
    }


def write_metric_table(results: OrderedDict[str, dict[str, float | int]]) -> None:
    metrics = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "mcc"]
    headings = ["Acc.", "Prec.", "Sens.", "Spec.", "F1", "ROC--AUC", "MCC"]
    maxima = {key: max(float(row[key]) for row in results.values()) for key in metrics}
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        "Model & " + " & ".join(headings) + r" \\",
        r"\midrule",
    ]
    for name, row in results.items():
        cells = []
        for key in metrics:
            value = float(row[key])
            rendered = f"{100 * value:.2f}\\%" if key != "mcc" else f"{value:.3f}"
            if np.isclose(value, maxima[key], atol=5e-5):
                rendered = rf"\textbf{{{rendered}}}"
            cells.append(rendered)
        lines.append(DISPLAY_NAMES[name] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (GENERATED_DIR / "model_metrics.tex").write_text("\n".join(lines), encoding="utf-8")


def verify_expected_results(results: OrderedDict[str, dict[str, float | int]]) -> None:
    expected = {
        "Custom KNN": [0.7273, 0.6250, 0.5556, 0.8200, 0.5882, 0.7900, 0.3869],
        "Decision Tree": [0.7857, 0.6981, 0.6852, 0.8400, 0.6916, 0.7887, 0.5275],
        "SVM": [0.7403, 0.6522, 0.5556, 0.8400, 0.6000, 0.7963, 0.4124],
        "Gaussian NB": [0.7013, 0.5667, 0.6296, 0.7400, 0.5965, 0.7646, 0.3617],
        "MLP": [0.7273, 0.6154, 0.5926, 0.8000, 0.6038, 0.8104, 0.3961],
    }
    metric_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc", "mcc"]
    for name, expected_values in expected.items():
        actual_values = [round(float(results[name][key]), 4) for key in metric_keys]
        if not np.allclose(actual_values, expected_values, atol=0.0001):
            raise RuntimeError(
                f"Unexpected metrics for {name}: {actual_values}; expected {expected_values}. "
                "The report must not be built from inconsistent results."
            )


def main() -> None:
    configure_plotting()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    x_data = df[FEATURES]
    y_data = df["Outcome"]

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.20,
        random_state=42,
        stratify=y_data,
    )

    preprocessor = make_preprocessor()
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_test_scaled = preprocessor.transform(x_test)

    models = build_models()
    results: OrderedDict[str, dict[str, float | int]] = OrderedDict()
    scores: OrderedDict[str, np.ndarray] = OrderedDict()
    fitted_models: dict[str, object] = {}

    for name, model in models.items():
        fitted_model = clone(model).fit(x_train_scaled, y_train)
        predictions = fitted_model.predict(x_test_scaled)
        probabilities = fitted_model.predict_proba(x_test_scaled)[:, 1]
        results[name] = evaluate(y_test, predictions, probabilities)
        scores[name] = probabilities
        fitted_models[name] = fitted_model

    verify_expected_results(results)
    importance = draw_permutation_importance(
        preprocessor,
        fitted_models["MLP"],
        x_test,
        y_test,
    )

    draw_workflow()
    draw_dataset_audit(df)
    draw_correlation_heatmap(df)
    draw_confusion_matrices(results)
    draw_model_comparison(results)
    draw_roc_curves(y_test, scores, results)
    write_metric_table(results)

    zero_counts = {column: int((df[column] == 0).sum()) for column in ZERO_AS_MISSING}
    training_medians = {
        column: float(x_train[column].replace(0, np.nan).median()) for column in ZERO_AS_MISSING
    }
    output = {
        "dataset": {
            "records": int(len(df)),
            "features": len(FEATURES),
            "duplicates": int(df.duplicated().sum()),
            "class_counts": {str(key): int(value) for key, value in y_data.value_counts().sort_index().items()},
            "zero_as_missing_counts": zero_counts,
        },
        "split": {
            "random_state": 42,
            "training_records": int(len(x_train)),
            "test_records": int(len(x_test)),
            "training_class_counts": {
                str(key): int(value) for key, value in y_train.value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(key): int(value) for key, value in y_test.value_counts().sort_index().items()
            },
        },
        "preprocessing": {
            "zero_as_missing_columns": ZERO_AS_MISSING,
            "training_medians": training_medians,
            "scaling": "StandardScaler fitted on the training set only",
        },
        "models": results,
        "mlp_permutation_importance": importance,
    }
    (GENERATED_DIR / "results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    metric_frame = pd.DataFrame(results).T
    metric_frame.to_csv(GENERATED_DIR / "model_metrics.csv", index_label="model")
    print(f"Generated {len(list(FIGURE_DIR.glob('*.pdf')))} figures and verified all report metrics.")


if __name__ == "__main__":
    main()
