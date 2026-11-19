from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uci_default_credit_card_clients_raw.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"


def main() -> None:

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {RAW_DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RAW_DATA_PATH)

    required_columns = {
        "ID",
        "X1",
        "X2",
        "X3",
        "X4",
        "X5",
        "Y",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    total_rows = len(data)
    total_columns = data.shape[1]

    complete_duplicates = int(
        data.duplicated().sum()
    )

    duplicated_id_rows = int(
        data.duplicated(
            subset=["ID"],
            keep=False,
        ).sum()
    )

    total_missing_values = int(
        data.isna().sum().sum()
    )

    target = data["Y"]

    invalid_target_mask = (
        target.notna()
        & ~target.isin([0, 1])
    )

    invalid_target_rows = int(
        invalid_target_mask.sum()
    )

    default_count = int(
        (target == 1).sum()
    )

    non_default_count = int(
        (target == 0).sum()
    )

    default_rate = float(
        target.mean()
    )

    # Los códigos sin documentar se cuentan aparte para revisar su recodificación.
    invalid_sex_mask = (
        data["X2"].notna()
        & ~data["X2"].isin([1, 2])
    )

    invalid_sex_codes = int(
        invalid_sex_mask.sum()
    )

    undocumented_education_mask = (
        data["X3"].notna()
        & ~data["X3"].isin([1, 2, 3, 4])
    )

    undocumented_education_codes = int(
        undocumented_education_mask.sum()
    )

    undocumented_marital_mask = (
        data["X4"].notna()
        & ~data["X4"].isin([1, 2, 3])
    )

    undocumented_marital_codes = int(
        undocumented_marital_mask.sum()
    )

    non_positive_credit_limit = int(
        (data["X1"] <= 0).sum()
    )

    implausible_age_mask = (
        (data["X5"] < 18)
        | (data["X5"] > 100)
    )

    implausible_age_rows = int(
        implausible_age_mask.sum()
    )

    payment_columns = [
        "X18",
        "X19",
        "X20",
        "X21",
        "X22",
        "X23",
    ]

    negative_payment_mask = (
        data[payment_columns] < 0
    ).any(axis=1)

    negative_payment_rows = int(
        negative_payment_mask.sum()
    )

    overview = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "total_columns",
                "complete_duplicate_rows",
                "duplicated_id_rows",
                "total_missing_values",
                "non_default_count",
                "default_count",
                "observed_default_rate",
            ],
            "value": [
                total_rows,
                total_columns,
                complete_duplicates,
                duplicated_id_rows,
                total_missing_values,
                non_default_count,
                default_count,
                default_rate,
            ],
        }
    )

    quality_checks = pd.DataFrame(
        {
            "check": [
                "Invalid target values",
                "Invalid sex codes",
                "Undocumented education codes",
                "Undocumented marital-status codes",
                "Non-positive credit limits",
                "Implausible ages",
                "Negative payment amounts",
            ],
            "affected_rows": [
                invalid_target_rows,
                invalid_sex_codes,
                undocumented_education_codes,
                undocumented_marital_codes,
                non_positive_credit_limit,
                implausible_age_rows,
                negative_payment_rows,
            ],
        }
    )

    column_profile = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": [
                str(data[column].dtype)
                for column in data.columns
            ],
            "missing_count": [
                int(data[column].isna().sum())
                for column in data.columns
            ],
            "unique_count": [
                int(data[column].nunique(dropna=False))
                for column in data.columns
            ],
        }
    )

    target_distribution = (
        target
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("target")
        .reset_index(name="count")
    )

    target_distribution["pct"] = (
        target_distribution["count"]
        / total_rows
        * 100
    )

    overview.to_csv(
        OUTPUT_DIR / "dataset_overview.csv",
        index=False,
    )

    quality_checks.to_csv(
        OUTPUT_DIR / "data_quality_checks.csv",
        index=False,
    )

    column_profile.to_csv(
        OUTPUT_DIR / "column_profile.csv",
        index=False,
    )

    target_distribution.to_csv(
        OUTPUT_DIR / "target_distribution.csv",
        index=False,
    )

    print()
    print("DATA QUALITY AUDIT")
    print("=" * 55)

    print(f"Rows: {total_rows:,}")
    print(f"Columns: {total_columns}")
    print(f"Complete duplicated rows: {complete_duplicates:,}")
    print(f"Duplicated ID rows: {duplicated_id_rows:,}")
    print(f"Total missing values: {total_missing_values:,}")

    print()
    print("TARGET")
    print("-" * 55)

    print(f"Non-default clients: {non_default_count:,}")
    print(f"Default clients: {default_count:,}")
    print(f"Observed default rate: {default_rate:.4%}")
    print(f"Invalid target values: {invalid_target_rows:,}")

    print()
    print("CATEGORICAL AND RANGE CHECKS")
    print("-" * 55)

    print(f"Invalid sex codes: {invalid_sex_codes:,}")
    print(
        "Undocumented education codes: "
        f"{undocumented_education_codes:,}"
    )
    print(
        "Undocumented marital-status codes: "
        f"{undocumented_marital_codes:,}"
    )
    print(
        "Non-positive credit limits: "
        f"{non_positive_credit_limit:,}"
    )
    print(f"Implausible ages: {implausible_age_rows:,}")
    print(
        "Rows with negative payment amounts: "
        f"{negative_payment_rows:,}"
    )

    print()
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()