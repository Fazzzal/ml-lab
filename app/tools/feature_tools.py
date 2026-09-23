import numpy as np
import pandas as pd

from app.tools.data_tools import (
    load_dataset,
    identify_identifier_columns,
)


TRANSFORMATIONS = {
    "numeric_interaction": {
        "description": "Multiply two numeric features.",
        "arity": 2,
        "input_types": ["numeric", "numeric"],
    },
    "numeric_ratio": {
        "description": "Divide one numeric feature by another.",
        "arity": 2,
        "input_types": ["numeric", "numeric"],
    },
    "numeric_difference": {
        "description": "Subtract the second numeric feature from the first.",
        "arity": 2,
        "input_types": ["numeric", "numeric"],
    },
    "numeric_sum": {
        "description": "Add two numeric features.",
        "arity": 2,
        "input_types": ["numeric", "numeric"],
    },
    "numeric_mean": {
        "description": "Calculate the mean of two numeric features.",
        "arity": 2,
        "input_types": ["numeric", "numeric"],
    },
    "numeric_square": {
        "description": "Square a numeric feature.",
        "arity": 1,
        "input_types": ["numeric"],
    },
    "numeric_log": {
        "description": "Apply log1p to the absolute value of a numeric feature.",
        "arity": 1,
        "input_types": ["numeric"],
    },
    "numeric_sqrt": {
        "description": "Apply square root to the absolute value of a numeric feature.",
        "arity": 1,
        "input_types": ["numeric"],
    },
    "numeric_abs": {
        "description": "Apply absolute value to a numeric feature.",
        "arity": 1,
        "input_types": ["numeric"],
    },
    "categorical_interaction": {
        "description": "Combine two categorical features into one categorical feature.",
        "arity": 2,
        "input_types": ["categorical", "categorical"],
    },
    "categorical_frequency": {
        "description": "Replace each categorical value with its frequency in the dataset.",
        "arity": 1,
        "input_types": ["categorical"],
    },
    "numeric_categorical_interaction": {
        "description": "Multiply a numeric feature by a frequency-encoded categorical feature.",
        "arity": 2,
        "input_types": ["numeric", "categorical"],
    },
}


def load_and_prepare_dataset(
    dataset_path: str,
    target_column: str,
) -> pd.DataFrame:
    return load_dataset(dataset_path)


def get_column_types(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, list[str]]:
    identifier_columns = identify_identifier_columns(df)

    columns = [
        column
        for column in df.columns
        if column != target_column
        and column not in identifier_columns
    ]

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in columns
        if column not in numeric_columns
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


def get_available_feature_types(
    column_names: list[str],
    dataframe: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> list[str]:
    if dataframe is None:
        return [
            name
            for name in TRANSFORMATIONS
        ]

    column_types = get_column_types(
        dataframe,
        target_column=target_column,
    )

    available = []

    for feature_type, specification in TRANSFORMATIONS.items():
        input_types = specification["input_types"]

        if all(
            len(column_types[input_type]) > 0
            for input_type in set(input_types)
        ):
            available.append(feature_type)

    return available


def get_feature_candidates(
    dataframe: pd.DataFrame,
    feature_type: str,
    target_column: str | None = None,
) -> list[list[str]]:
    column_types = get_column_types(
        dataframe,
        target_column=target_column,
    )

    specification = TRANSFORMATIONS.get(feature_type)

    if specification is None:
        raise ValueError(
            f"Unsupported feature type: {feature_type}"
        )

    input_types = specification["input_types"]

    if len(input_types) == 1:
        return [
            [column]
            for column in column_types[input_types[0]]
        ]

    if len(input_types) == 2:
        first_type = input_types[0]
        second_type = input_types[1]

        candidates = []

        for first_column in column_types[first_type]:
            for second_column in column_types[second_type]:
                if (
                    first_type == second_type
                    and first_column == second_column
                ):
                    continue

                candidates.append([
                    first_column,
                    second_column,
                ])

        return candidates

    raise ValueError(
        f"Unsupported feature arity for {feature_type}"
    )


def create_feature(
    df: pd.DataFrame,
    feature_name: str,
    feature_type: str,
    source_columns: list[str],
) -> pd.DataFrame:
    if feature_type not in TRANSFORMATIONS:
        raise ValueError(
            f"Unsupported feature type: {feature_type}"
        )

    result = df.copy()

    if feature_type == "numeric_interaction":
        first, second = source_columns

        result[feature_name] = (
            pd.to_numeric(result[first], errors="coerce")
            * pd.to_numeric(result[second], errors="coerce")
        )

    elif feature_type == "numeric_ratio":
        first, second = source_columns

        denominator = pd.to_numeric(
            result[second],
            errors="coerce",
        ).replace(0, np.nan)

        numerator = pd.to_numeric(
            result[first],
            errors="coerce",
        )

        result[feature_name] = numerator / denominator

    elif feature_type == "numeric_difference":
        first, second = source_columns

        result[feature_name] = (
            pd.to_numeric(result[first], errors="coerce")
            - pd.to_numeric(result[second], errors="coerce")
        )

    elif feature_type == "numeric_sum":
        first, second = source_columns

        result[feature_name] = (
            pd.to_numeric(result[first], errors="coerce")
            + pd.to_numeric(result[second], errors="coerce")
        )

    elif feature_type == "numeric_mean":
        first, second = source_columns

        result[feature_name] = (
            pd.to_numeric(result[first], errors="coerce")
            + pd.to_numeric(result[second], errors="coerce")
        ) / 2.0

    elif feature_type == "numeric_square":
        source = source_columns[0]

        values = pd.to_numeric(
            result[source],
            errors="coerce",
        )

        result[feature_name] = values ** 2

    elif feature_type == "numeric_log":
        source = source_columns[0]

        values = pd.to_numeric(
            result[source],
            errors="coerce",
        )

        result[feature_name] = np.log1p(np.abs(values))

    elif feature_type == "numeric_sqrt":
        source = source_columns[0]

        values = pd.to_numeric(
            result[source],
            errors="coerce",
        )

        result[feature_name] = np.sqrt(np.abs(values))

    elif feature_type == "numeric_abs":
        source = source_columns[0]

        values = pd.to_numeric(
            result[source],
            errors="coerce",
        )

        result[feature_name] = np.abs(values)

    elif feature_type == "categorical_interaction":
        first, second = source_columns

        result[feature_name] = (
            result[first].astype(str)
            + "__"
            + result[second].astype(str)
        )

    elif feature_type == "categorical_frequency":
        source = source_columns[0]

        frequencies = (
            result[source]
            .value_counts(normalize=True)
        )

        result[feature_name] = (
            result[source]
            .map(frequencies)
        )

    elif feature_type == "numeric_categorical_interaction":
        numeric_column, categorical_column = source_columns

        frequencies = (
            result[categorical_column]
            .value_counts(normalize=True)
        )

        encoded = result[categorical_column].map(
            frequencies
        )

        numeric_values = pd.to_numeric(
            result[numeric_column],
            errors="coerce",
        )

        result[feature_name] = (
            numeric_values * encoded
        )

    else:
        raise ValueError(
            f"Unsupported feature type: {feature_type}"
        )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


def apply_feature_engineering(
    dataset_path: str,
    target_column: str,
    feature_type: str,
    feature_name: str,
    source_columns: list[str],
) -> pd.DataFrame:
    df = load_and_prepare_dataset(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    column_types = get_column_types(
        df,
        target_column=target_column,
    )

    specification = TRANSFORMATIONS.get(feature_type)

    if specification is None:
        raise ValueError(
            f"Unsupported feature type: {feature_type}"
        )

    if len(source_columns) != len(
        specification["input_types"]
    ):
        raise ValueError(
            f"Feature type '{feature_type}' expects "
            f"{len(specification['input_types'])} source "
            f"columns, received {len(source_columns)}."
        )

    for column, expected_type in zip(
        source_columns,
        specification["input_types"],
    ):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        if column == target_column:
            raise ValueError(
                "Target column cannot be used as a source feature."
            )

        if column not in column_types[expected_type]:
            raise ValueError(
                f"Column '{column}' is not "
                f"{expected_type}."
            )

    return create_feature(
        df=df,
        feature_name=feature_name,
        feature_type=feature_type,
        source_columns=source_columns,
    )