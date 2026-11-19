from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve


# =========================================================
# 1. RUTAS DEL PROYECTO
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_data.csv"
)

PIPELINE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "final_pd_pipeline.joblib"
)

DECILE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_test_decile_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


# =========================================================
# 2. VARIABLES DEL MODELO FINAL
# =========================================================

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


# =========================================================
# 3. FUNCIÓN PRINCIPAL
# =========================================================

def main() -> None:
    """Create final model performance figures."""

    required_paths = [
        TEST_DATA_PATH,
        PIPELINE_PATH,
        DECILE_SUMMARY_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {path}"
            )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_data = pd.read_csv(
        TEST_DATA_PATH
    )

    decile_summary = pd.read_csv(
        DECILE_SUMMARY_PATH
    )

    final_pipeline = joblib.load(
        PIPELINE_PATH
    )

    X_test = test_data[
        FINAL_FEATURES
    ].copy()

    y_test = test_data[
        "default"
    ].copy()

    test_pd = final_pipeline.predict_proba(
        X_test
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        test_pd
    )

    # =====================================================
    # 4. CURVA ROC
    # =====================================================

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            test_pd
        )
    )

    roc_path = (
        FIGURES_DIR
        / "final_test_roc_curve.png"
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Logistic model — AUC = {test_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Final PD Model")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        roc_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 5. DEFAULT OBSERVADO POR DECIL
    # =====================================================

    decile_risk_path = (
        FIGURES_DIR
        / "final_test_default_rate_by_decile.png"
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        decile_summary["risk_decile"],
        decile_summary["observed_default_rate_pct"],
        marker="o"
    )

    plt.xlabel("Risk Decile")
    plt.ylabel("Observed Default Rate (%)")
    plt.title("Observed Default Rate by Risk Decile")
    plt.xticks(
        decile_summary["risk_decile"]
    )
    plt.tight_layout()

    plt.savefig(
        decile_risk_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 6. CALIBRACIÓN POR DECILES
    # =====================================================

    calibration_path = (
        FIGURES_DIR
        / "final_test_calibration_by_decile.png"
    )

    maximum_rate = max(
        decile_summary[
            "observed_default_rate_pct"
        ].max(),
        decile_summary[
            "average_predicted_pd_pct"
        ].max()
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, maximum_rate],
        [0, maximum_rate],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        decile_summary["average_predicted_pd_pct"],
        decile_summary["observed_default_rate_pct"],
        marker="o",
        label="Final PD model"
    )

    plt.xlabel("Average Predicted PD (%)")
    plt.ylabel("Observed Default Rate (%)")
    plt.title("Calibration by Risk Decile")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        calibration_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 7. MOSTRAR RESULTADOS
    # =====================================================

    print()
    print("FINAL MODEL FIGURES")
    print("=" * 55)
    print(f"Test AUC: {test_auc:.6f}")
    print()
    print(f"ROC curve saved to: {roc_path}")
    print(
        "Default rate by decile saved to: "
        f"{decile_risk_path}"
    )
    print(
        "Calibration chart saved to: "
        f"{calibration_path}"
    )


if __name__ == "__main__":
    main()