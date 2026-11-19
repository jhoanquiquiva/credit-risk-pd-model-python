from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

PIPELINE_PATH = PROJECT_ROOT / "outputs" / "models" / "final_pd_pipeline.joblib"

TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_test_metrics.csv"

TEST_DECILES_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_test_decile_summary.csv"

PSI_BANDS_PATH = PROJECT_ROOT / "outputs" / "tables" / "score_psi_by_band.csv"

PSI_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "score_psi_summary.csv"


FINAL_FEATURES = [
    "sex",
    "education",
    "marital_status",
    "has_severe_recent_delay",
    "has_negative_utilization",
    "has_extreme_utilization",
    "credit_limit",
    "age",
    "months_with_severe_delay",
    "avg_payment_amount",
    "avg_credit_utilization",
    "months_no_payment_with_positive_bill",
]


def calculate_ks(
    y_true: pd.Series,
    predicted_pd: np.ndarray,
) -> dict:
    """Calculate maximum KS and its associated threshold."""

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_true,
            predicted_pd,
        )
    )

    ks_values = (
        true_positive_rate
        - false_positive_rate
    )

    max_position = int(
        np.argmax(ks_values)
    )

    return {
        "ks": float(ks_values[max_position]),
        "threshold": float(thresholds[max_position]),
        "tpr": float(true_positive_rate[max_position]),
        "fpr": float(false_positive_rate[max_position]),
    }


def main() -> None:
    """Evaluate the final saved PD pipeline."""

    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        PIPELINE_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {path}"
            )

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    final_pipeline = joblib.load(
        PIPELINE_PATH
    )

    X_train = train_data[
        FINAL_FEATURES
    ].copy()

    y_train = train_data[
        "default"
    ].copy()

    X_test = test_data[
        FINAL_FEATURES
    ].copy()

    y_test = test_data[
        "default"
    ].copy()

    train_pd = final_pipeline.predict_proba(
        X_train
    )[:, 1]

    test_pd = final_pipeline.predict_proba(
        X_test
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        test_pd,
    )

    test_gini = (
        2 * test_auc - 1
    )

    test_ks = calculate_ks(
        y_test,
        test_pd,
    )

    observed_default_rate = float(
        y_test.mean()
    )

    average_predicted_pd = float(
        test_pd.mean()
    )

    oe_ratio = (
        observed_default_rate
        / average_predicted_pd
    )

    test_brier = brier_score_loss(
        y_test,
        test_pd,
    )

    constant_test_pd = np.full(
        shape=len(y_test),
        fill_value=y_train.mean(),
    )

    baseline_brier = brier_score_loss(
        y_test,
        constant_test_pd,
    )

    brier_skill_score = (
        1
        - test_brier
        / baseline_brier
    )

    final_test_metrics = pd.DataFrame(
        {
            "metric": [
                "test_clients",
                "test_defaults",
                "auc",
                "gini",
                "ks",
                "ks_threshold",
                "ks_tpr",
                "ks_fpr",
                "observed_default_rate",
                "average_predicted_pd",
                "observed_to_expected_ratio",
                "brier_score",
                "baseline_brier_score",
                "brier_skill_score",
            ],
            "value": [
                len(test_data),
                int(y_test.sum()),
                test_auc,
                test_gini,
                test_ks["ks"],
                test_ks["threshold"],
                test_ks["tpr"],
                test_ks["fpr"],
                observed_default_rate,
                average_predicted_pd,
                oe_ratio,
                test_brier,
                baseline_brier,
                brier_skill_score,
            ],
        }
    )

    # Ordenar por probabilidad estimada para comparar lo observado dentro de cada decil.
    test_predictions = pd.DataFrame(
        {
            "observed_default": y_test.to_numpy(),
            "predicted_pd": test_pd,
        }
    )

    test_predictions["risk_decile"] = (
        pd.qcut(
            test_predictions["predicted_pd"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    test_decile_summary = (
        test_predictions
        .groupby(
            "risk_decile",
            observed=True,
        )
        .agg(
            clients=(
                "observed_default",
                "count",
            ),
            defaults=(
                "observed_default",
                "sum",
            ),
            observed_default_rate=(
                "observed_default",
                "mean",
            ),
            average_predicted_pd=(
                "predicted_pd",
                "mean",
            ),
        )
        .reset_index()
    )

    test_decile_summary[
        "observed_default_rate_pct"
    ] = (
        test_decile_summary[
            "observed_default_rate"
        ]
        * 100
    )

    test_decile_summary[
        "average_predicted_pd_pct"
    ] = (
        test_decile_summary[
            "average_predicted_pd"
        ]
        * 100
    )

    # Usar las bandas de entrenamiento en las dos muestras hace comparables sus frecuencias.
    train_score = pd.Series(
        train_pd,
        name="predicted_pd",
    )

    test_score = pd.Series(
        test_pd,
        name="predicted_pd",
    )

    _, score_bins = pd.qcut(
        train_score,
        q=10,
        retbins=True,
        duplicates="drop",
    )

    score_bins[0] = -np.inf
    score_bins[-1] = np.inf

    train_score_band = pd.cut(
        train_score,
        bins=score_bins,
        include_lowest=True,
    )

    test_score_band = pd.cut(
        test_score,
        bins=score_bins,
        include_lowest=True,
    )

    train_distribution = (
        train_score_band
        .value_counts(
            sort=False,
            normalize=True,
        )
    )

    test_distribution = (
        test_score_band
        .value_counts(
            sort=False,
            normalize=True,
        )
        .reindex(
            train_distribution.index,
            fill_value=0,
        )
    )

    epsilon = 1e-6

    psi_table = pd.DataFrame(
        {
            "score_band": (
                train_distribution
                .index
                .astype(str)
            ),
            "train_pct": (
                train_distribution.values
            ),
            "test_pct": (
                test_distribution.values
            ),
        }
    )

    psi_table["train_pct_adjusted"] = (
        psi_table["train_pct"]
        .clip(lower=epsilon)
    )

    psi_table["test_pct_adjusted"] = (
        psi_table["test_pct"]
        .clip(lower=epsilon)
    )

    psi_table["psi_component"] = (
        (
            psi_table["test_pct_adjusted"]
            - psi_table["train_pct_adjusted"]
        )
        * np.log(
            psi_table["test_pct_adjusted"]
            / psi_table["train_pct_adjusted"]
        )
    )

    psi_total = float(
        psi_table["psi_component"].sum()
    )

    if psi_total < 0.10:
        psi_interpretation = (
            "small_population_shift"
        )
    elif psi_total < 0.25:
        psi_interpretation = (
            "moderate_population_shift"
        )
    else:
        psi_interpretation = (
            "material_population_shift"
        )

    psi_summary = pd.DataFrame(
        {
            "comparison": [
                "train_vs_test"
            ],
            "psi_total": [
                psi_total
            ],
            "interpretation": [
                psi_interpretation
            ],
            "limitation": [
                (
                    "Random split; this is not "
                    "an out-of-time stability test."
                )
            ],
        }
    )

    TEST_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_test_metrics.to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    test_decile_summary.to_csv(
        TEST_DECILES_PATH,
        index=False,
    )

    psi_table.to_csv(
        PSI_BANDS_PATH,
        index=False,
    )

    psi_summary.to_csv(
        PSI_SUMMARY_PATH,
        index=False,
    )

    print()
    print("FINAL PD MODEL TEST EVALUATION")
    print("=" * 55)

    print(f"Test clients: {len(test_data):,}")
    print(f"Test defaults: {int(y_test.sum()):,}")
    print()

    print(f"Test AUC: {test_auc:.6f}")
    print(f"Test Gini: {test_gini:.6f}")
    print(f"Test KS: {test_ks['ks']:.6f}")
    print(
        "KS threshold: "
        f"{test_ks['threshold']:.6f}"
    )
    print()

    print(
        "Observed default rate: "
        f"{observed_default_rate:.4%}"
    )

    print(
        "Average predicted PD: "
        f"{average_predicted_pd:.4%}"
    )

    print(f"O/E ratio: {oe_ratio:.6f}")
    print(f"Brier Score: {test_brier:.6f}")
    print(
        "Brier Skill Score: "
        f"{brier_skill_score:.4%}"
    )
    print()

    print(f"Score PSI: {psi_total:.6f}")
    print(
        "PSI interpretation: "
        f"{psi_interpretation}"
    )
    print()

    print(f"Metrics saved to: {TEST_METRICS_PATH}")
    print(f"Deciles saved to: {TEST_DECILES_PATH}")
    print(f"PSI bands saved to: {PSI_BANDS_PATH}")
    print(f"PSI summary saved to: {PSI_SUMMARY_PATH}")


if __name__ == "__main__":
    main()