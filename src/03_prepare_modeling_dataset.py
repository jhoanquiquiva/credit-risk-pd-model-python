from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uci_default_credit_card_clients_raw.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


RENAME_MAP = {
    "ID": "client_id",
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education_code_original",
    "X4": "marital_status_code_original",
    "X5": "age",

    "X6": "repayment_status_m1",
    "X7": "repayment_status_m2",
    "X8": "repayment_status_m3",
    "X9": "repayment_status_m4",
    "X10": "repayment_status_m5",
    "X11": "repayment_status_m6",

    "X12": "bill_amount_m1",
    "X13": "bill_amount_m2",
    "X14": "bill_amount_m3",
    "X15": "bill_amount_m4",
    "X16": "bill_amount_m5",
    "X17": "bill_amount_m6",

    "X18": "payment_amount_m1",
    "X19": "payment_amount_m2",
    "X20": "payment_amount_m3",
    "X21": "payment_amount_m4",
    "X22": "payment_amount_m5",
    "X23": "payment_amount_m6",

    "Y": "default",
}


REPAYMENT_COLUMNS = [
    "repayment_status_m1",
    "repayment_status_m2",
    "repayment_status_m3",
    "repayment_status_m4",
    "repayment_status_m5",
    "repayment_status_m6",
]

BILL_COLUMNS = [
    "bill_amount_m1",
    "bill_amount_m2",
    "bill_amount_m3",
    "bill_amount_m4",
    "bill_amount_m5",
    "bill_amount_m6",
]

PAYMENT_COLUMNS = [
    "payment_amount_m1",
    "payment_amount_m2",
    "payment_amount_m3",
    "payment_amount_m4",
    "payment_amount_m5",
    "payment_amount_m6",
]


CANDIDATE_FEATURES = [
    "credit_limit",
    "sex",
    "education",
    "marital_status",
    "age",

    "has_recent_delay",
    "has_severe_recent_delay",
    "months_with_delay",
    "months_with_severe_delay",
    "max_repayment_delay",

    "avg_bill_amount",
    "avg_payment_amount",

    "avg_credit_utilization",
    "has_negative_utilization",
    "is_over_limit",
    "has_extreme_utilization",

    "months_no_payment_with_positive_bill",
]


def main() -> None:
    """Create reproducible modeling, train, validation and test datasets."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el dataset crudo. "
            "Ejecuta primero src/01_download_uci_data.py. "
            f"Ruta esperada: {RAW_DATA_PATH}"
        )

    data = pd.read_csv(RAW_DATA_PATH)

    expected_columns = {
        "ID",
        *[f"X{i}" for i in range(1, 24)],
        "Y",
    }

    missing_original_columns = (
        expected_columns - set(data.columns)
    )

    if missing_original_columns:
        raise ValueError(
            "Faltan columnas originales: "
            f"{sorted(missing_original_columns)}"
        )

    # Conservar los códigos originales permite comprobar qué categorías se agruparon.
    model_data = data.rename(
        columns=RENAME_MAP
    ).copy()

    model_data["education"] = (
        model_data["education_code_original"]
        .replace(
            {
                0: 4,
                5: 4,
                6: 4,
            }
        )
    )

    model_data["marital_status"] = (
        model_data["marital_status_code_original"]
        .replace(
            {
                0: 3,
            }
        )
    )

    delay_flags = (
        model_data[REPAYMENT_COLUMNS] > 0
    )

    severe_delay_flags = (
        model_data[REPAYMENT_COLUMNS] >= 2
    )

    model_data["has_recent_delay"] = (
        model_data["repayment_status_m1"] > 0
    ).astype(int)

    model_data["has_severe_recent_delay"] = (
        model_data["repayment_status_m1"] >= 2
    ).astype(int)

    model_data["months_with_delay"] = (
        delay_flags.sum(axis=1)
    )

    model_data["months_with_severe_delay"] = (
        severe_delay_flags.sum(axis=1)
    )

    model_data["max_repayment_delay"] = (
        model_data[REPAYMENT_COLUMNS]
        .max(axis=1)
    )

    model_data["avg_bill_amount"] = (
        model_data[BILL_COLUMNS]
        .mean(axis=1)
    )

    model_data["avg_payment_amount"] = (
        model_data[PAYMENT_COLUMNS]
        .mean(axis=1)
    )

    model_data["avg_credit_utilization_raw"] = (
        model_data["avg_bill_amount"]
        / model_data["credit_limit"]
    )

    # Contar un mes sin pago solo cuando hubo una factura positiva.
    no_payment_flags = pd.DataFrame(
        {
            f"m{month}": (
                (
                    model_data[
                        f"bill_amount_m{month}"
                    ] > 0
                )
                & (
                    model_data[
                        f"payment_amount_m{month}"
                    ] == 0
                )
            )
            for month in range(1, 7)
        },
        index=model_data.index,
    )

    model_data[
        "months_no_payment_with_positive_bill"
    ] = no_payment_flags.sum(axis=1)

    # Dividir antes de estimar el límite de utilización evita usar información de prueba.
    development_index, test_index = train_test_split(
        model_data.index,
        test_size=0.20,
        stratify=model_data["default"],
        random_state=42,
    )

    train_index, validation_index = train_test_split(
        development_index,
        test_size=0.25,
        stratify=model_data.loc[
            development_index,
            "default",
        ],
        random_state=42,
    )

    utilization_cap = float(
        model_data.loc[
            train_index,
            "avg_credit_utilization_raw",
        ].quantile(0.99)
    )

    raw_utilization = model_data[
        "avg_credit_utilization_raw"
    ]

    model_data["has_negative_utilization"] = (
        raw_utilization < 0
    ).astype(int)

    model_data["is_over_limit"] = (
        raw_utilization > 1
    ).astype(int)

    model_data["has_extreme_utilization"] = (
        raw_utilization > utilization_cap
    ).astype(int)

    model_data["avg_credit_utilization"] = (
        raw_utilization.clip(
            lower=0,
            upper=utilization_cap,
        )
    )

    modeling_dataset = model_data[
        ["client_id", "default"]
        + CANDIDATE_FEATURES
    ].copy()

    train_data = (
        modeling_dataset
        .loc[train_index]
        .copy()
    )

    validation_data = (
        modeling_dataset
        .loc[validation_index]
        .copy()
    )

    test_data = (
        modeling_dataset
        .loc[test_index]
        .copy()
    )

    total_rows = len(modeling_dataset)
    distinct_clients = (
        modeling_dataset["client_id"].nunique()
    )

    duplicate_clients = int(
        modeling_dataset["client_id"]
        .duplicated()
        .sum()
    )

    missing_values = int(
        modeling_dataset.isna().sum().sum()
    )

    if total_rows != 30000:
        raise ValueError(
            f"Se esperaban 30.000 filas y se obtuvieron {total_rows}."
        )

    if distinct_clients != total_rows:
        raise ValueError(
            "La base final no mantiene una fila por cliente."
        )

    if duplicate_clients != 0:
        raise ValueError(
            f"Se encontraron {duplicate_clients} clientes duplicados."
        )

    if missing_values != 0:
        raise ValueError(
            f"Se encontraron {missing_values} valores faltantes."
        )

    if (
        len(train_data)
        + len(validation_data)
        + len(test_data)
        != total_rows
    ):
        raise ValueError(
            "Las muestras no contienen el total de clientes."
        )

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "total_columns",
                "distinct_clients",
                "duplicate_clients",
                "missing_values",
                "observed_defaults",
                "observed_default_rate",
                "utilization_cap_train_p99",
            ],
            "value": [
                total_rows,
                modeling_dataset.shape[1],
                distinct_clients,
                duplicate_clients,
                missing_values,
                int(modeling_dataset["default"].sum()),
                float(modeling_dataset["default"].mean()),
                utilization_cap,
            ],
        }
    )

    split_summary = pd.DataFrame(
        {
            "sample": [
                "train",
                "validation",
                "test",
            ],
            "clients": [
                len(train_data),
                len(validation_data),
                len(test_data),
            ],
            "defaults": [
                int(train_data["default"].sum()),
                int(validation_data["default"].sum()),
                int(test_data["default"].sum()),
            ],
            "default_rate": [
                train_data["default"].mean(),
                validation_data["default"].mean(),
                test_data["default"].mean(),
            ],
        }
    )

    split_summary["default_rate_pct"] = (
        split_summary["default_rate"] * 100
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling_dataset.to_csv(
        PROCESSED_DIR / "modeling_dataset.csv",
        index=False,
    )

    train_data.to_csv(
        PROCESSED_DIR / "train_data.csv",
        index=False,
    )

    validation_data.to_csv(
        PROCESSED_DIR / "validation_data.csv",
        index=False,
    )

    test_data.to_csv(
        PROCESSED_DIR / "test_data.csv",
        index=False,
    )

    validation_summary.to_csv(
        TABLES_DIR
        / "modeling_dataset_validation.csv",
        index=False,
    )

    split_summary.to_csv(
        TABLES_DIR
        / "sample_split_summary.csv",
        index=False,
    )

    print()
    print("MODELING DATASET PREPARATION")
    print("=" * 55)
    print(f"Rows: {total_rows:,}")
    print(
        f"Distinct clients: "
        f"{distinct_clients:,}"
    )
    print(
        f"Duplicate clients: "
        f"{duplicate_clients:,}"
    )
    print(
        f"Missing values: "
        f"{missing_values:,}"
    )
    print(
        "Observed default rate: "
        f"{modeling_dataset['default'].mean():.4%}"
    )
    print(
        "Training utilization cap (P99): "
        f"{utilization_cap:.6f}"
    )
    print()
    print(split_summary.to_string(index=False))
    print()
    print(
        "Processed datasets saved to: "
        f"{PROCESSED_DIR}"
    )
    print(
        "Validation tables saved to: "
        f"{TABLES_DIR}"
    )


if __name__ == "__main__":
    main()