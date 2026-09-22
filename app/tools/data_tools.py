import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


IDENTIFIER_NAMES = {
    "customerid",
    "userid",
    "accountid",
    "recordid",
}


def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    for column in df.columns:
        normalized = column.strip().lower()

        if normalized in {
            "totalcharges",
            "total_charge",
            "total_charge_amount",
        }:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def identify_identifier_columns(
    df: pd.DataFrame,
) -> list[str]:
    identifiers = []

    for column in df.columns:
        normalized = column.strip().lower()

        if normalized in IDENTIFIER_NAMES:
            identifiers.append(column)

    return identifiers


def profile_dataset(
    dataset_path: str,
    target_column: str | None = None,
) -> dict:
    df = load_dataset(dataset_path)

    if target_column is None:
        target_column = df.columns[-1]

    identifier_columns = identify_identifier_columns(df)

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
        and column not in identifier_columns
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    missing_columns = [
        column
        for column in df.columns
        if df[column].isna().any()
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),

        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,

        "missing_values": {
            column: int(df[column].isna().sum())
            for column in df.columns
            if df[column].isna().any()
        },

        "duplicates": int(df.duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),

        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "identifier_columns": identifier_columns,

        "target_column": target_column,
        "target_dtype": str(df[target_column].dtype),
    }

def load_and_profile(
    dataset_path: str,
    target_column: str,
) -> tuple[pd.DataFrame, dict]:
    df = load_dataset(dataset_path)

    profile = profile_dataset(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    return df, profile


def split_dataset_indices(
    dataset_path: str,
    target_column: str,
    problem_type: str | None = None,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    if not np.isclose(
        train_size + val_size + test_size,
        1.0,
    ):
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0"
        )

    df = load_dataset(dataset_path)

    indices = np.arange(len(df))

    stratify = None

    if problem_type == "classification":
        target = df[target_column]

        if target.nunique() > 1:
            stratify = target

    train_indices, remaining_indices = train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_test_size = test_size / (
        val_size + test_size
    )

    remaining_target = None

    if stratify is not None:
        remaining_target = df.iloc[
            remaining_indices
        ][target_column]

    val_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=remaining_target,
    )

    return {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }