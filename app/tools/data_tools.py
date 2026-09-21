import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def profile_dataset(path: str) -> dict:
    df = load_dataset(path)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].to_dict()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing_values,
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }
    }


def split_dataset_indices(
    dataset_path: str,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Compute a fixed 3-way split, once, as row-position indices
    into the original CSV (0-based, matching df.iloc / df.reset_index
    order). Feature engineering only adds columns and never
    reorders/drops rows, so these indices stay valid across every
    experiment run on this dataset, engineered or not.

    Stratified on the target for classification problems.
    """

    if not (0 < test_size < 1) or not (0 < val_size < 1):
        raise ValueError(
            "test_size and val_size must each be between 0 and 1"
        )

    if test_size + val_size >= 1:
        raise ValueError(
            "test_size + val_size must be less than 1 so a "
            "training split remains"
        )

    df = load_dataset(dataset_path)
    indices = np.arange(len(df))

    y = df[target_column] if problem_type == "classification" else None

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y is not None else None,
    )

    relative_val_size = val_size / (1 - test_size)

    stratify_train_val = (
        y.iloc[train_val_idx] if y is not None else None
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
