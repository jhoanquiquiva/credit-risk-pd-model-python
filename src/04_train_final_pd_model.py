from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"

VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation_data.csv"

MODEL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models" / "final_pd_pipeline.joblib"

METRICS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_model_validation_metrics.csv"

COEFFICIENTS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_model_coefficients.csv"


CATEGORICAL_FEATURES = [
    "sex",
    "education",
    "marital_status",
]

BINARY_FEATURES = [
    "has_severe_recent_delay",
    "has_negative_utilization",
    "has_extreme_utilization",
]

NUMERIC_FEATURES = [
    "credit_limit",
    "age",
    "months_with_severe_delay",
    "avg_payment_amount",
    "avg_credit_utilization",
    "months_no_payment_with_positive_bill",
]

FINAL_FEATURES = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + NUMERIC_FEATURES
)


def main() -> None:
    """Train and save the final parsimonious PD model."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró entrenamiento: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró validación: {VALIDATION_PATH}"
        )

    train_data = pd.read_csv(TRAIN_PATH)
    validation_data = pd.read_csv(VALIDATION_PATH)

    missing_train_features = (
        set(FINAL_FEATURES) - set(train_data.columns)
    )

    if missing_train_features:
        raise ValueError(
            "Faltan variables en entrenamiento: "
            f"{sorted(missing_train_features)}"
        )

    X_train = train_data[FINAL_FEATURES].copy()
    y_train = train_data["default"].copy()

    X_validation = validation_data[FINAL_FEATURES].copy()
    y_validation = validation_data["default"].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
        ]
    )

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    final_pipeline.fit(
        X_train,
        y_train,
    )

    validation_pd = final_pipeline.predict_proba(
        X_validation
    )[:, 1]

    validation_auc = roc_auc_score(
        y_validation,
        validation_pd,
    )

    validation_gini = (
        2 * validation_auc - 1
    )

    observed_default_rate = float(
        y_validation.mean()
    )

    average_predicted_pd = float(
        validation_pd.mean()
    )

    oe_ratio = (
        observed_default_rate
        / average_predicted_pd
    )

    validation_brier = brier_score_loss(
        y_validation,
        validation_pd,
    )

    naive_pd = np.full(
        shape=len(y_validation),
        fill_value=y_train.mean(),
    )

    naive_brier = brier_score_loss(
        y_validation,
        naive_pd,
    )

    brier_skill_score = (
        1
        - validation_brier
        / naive_brier
    )

    metrics = pd.DataFrame(
        {
            "metric": [
                "training_clients",
                "validation_clients",
                "original_feature_count",
                "validation_auc",
                "validation_gini",
                "observed_default_rate",
                "average_predicted_pd",
                "observed_to_expected_ratio",
                "validation_brier_score",
                "brier_skill_score",
            ],
            "value": [
                len(train_data),
                len(validation_data),
                len(FINAL_FEATURES),
                validation_auc,
                validation_gini,
                observed_default_rate,
                average_predicted_pd,
                oe_ratio,
                validation_brier,
                brier_skill_score,
            ],
        }
    )

    # La codificación de categorías cambia los nombres y el número de coeficientes.
    fitted_preprocessor = (
        final_pipeline.named_steps["preprocessor"]
    )

    fitted_model = (
        final_pipeline.named_steps["model"]
    )

    transformed_feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    coefficients = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": fitted_model.coef_[0],
        }
    )

    coefficients["odds_ratio"] = np.exp(
        coefficients["coefficient"]
    )

    coefficients = coefficients.sort_values(
        by="coefficient",
        ascending=False,
    )

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_pipeline,
        MODEL_OUTPUT_PATH,
    )

    metrics.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_OUTPUT_PATH,
        index=False,
    )

    print()
    print("FINAL PD MODEL TRAINING")
    print("=" * 55)
    print(f"Training clients: {len(train_data):,}")
    print(f"Validation clients: {len(validation_data):,}")
    print(f"Original features: {len(FINAL_FEATURES)}")
    print()
    print(f"Validation AUC: {validation_auc:.6f}")
    print(f"Validation Gini: {validation_gini:.6f}")
    print(f"Observed default rate: {observed_default_rate:.4%}")
    print(f"Average predicted PD: {average_predicted_pd:.4%}")
    print(f"O/E ratio: {oe_ratio:.6f}")
    print(f"Brier Score: {validation_brier:.6f}")
    print(f"Brier Skill Score: {brier_skill_score:.4%}")
    print()
    print(f"Pipeline saved to: {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_OUTPUT_PATH}")
    print(f"Coefficients saved to: {COEFFICIENTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()