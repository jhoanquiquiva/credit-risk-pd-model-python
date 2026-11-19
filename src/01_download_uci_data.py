from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Download and save the original UCI credit default dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI dataset 350...")
    dataset = fetch_ucirepo(id=350)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    data_parts = []

    # Preserve the original client identifier when supplied separately.
    ids = getattr(dataset.data, "ids", None)

    if ids is not None:
        data_parts.append(ids.copy())

    data_parts.extend([features, targets])

    raw_data = pd.concat(data_parts, axis=1)

    raw_data_path = (
        RAW_DATA_DIR
        / "uci_default_credit_card_clients_raw.csv"
    )

    variable_dictionary_path = (
        RAW_DATA_DIR
        / "uci_variable_dictionary.csv"
    )

    raw_data.to_csv(
        raw_data_path,
        index=False
    )

    dataset.variables.to_csv(
        variable_dictionary_path,
        index=False
    )

    target = pd.to_numeric(
        targets.iloc[:, 0],
        errors="raise"
    )

    print()
    print("Dataset downloaded successfully.")
    print(f"Rows: {raw_data.shape[0]:,}")
    print(f"Columns: {raw_data.shape[1]}")
    print(f"Duplicated complete rows: {raw_data.duplicated().sum():,}")
    print()

    print("Target distribution:")
    print(
        target
        .value_counts(dropna=False)
        .sort_index()
    )

    print()
    print(f"Observed default rate: {target.mean():.4%}")
    print()
    print(f"Raw data saved to: {raw_data_path}")
    print(f"Variable dictionary saved to: {variable_dictionary_path}")


if __name__ == "__main__":
    main()
