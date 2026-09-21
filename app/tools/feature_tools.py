import pandas as pd


# Columns each transform needs. Transforms marked with an empty
# "required" set instead need at least one column from their
# "any_of" list (a partial signal is still useful, e.g. only some
# streaming columns present).
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SUPPORT_SECURITY_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

STREAMING_COLUMNS = [
    "StreamingTV",
    "StreamingMovies",
]

FEATURE_REQUIREMENTS = {
    "total_charges_per_tenure": {
        "required": {"TotalCharges", "tenure"},
        "any_of": [],
    },
    "monthly_charge_tenure": {
        "required": {"MonthlyCharges", "tenure"},
        "any_of": [],
    },
    "service_count": {
        "required": set(),
        "any_of": SERVICE_COLUMNS,
    },
    "support_security_count": {
        "required": set(),
        "any_of": SUPPORT_SECURITY_COLUMNS,
    },
    "has_streaming": {
        "required": set(),
        "any_of": STREAMING_COLUMNS,
    },
}


def get_available_feature_types(column_names: list[str]) -> list[str]:
    """Return only the feature_type values that this dataset can
    actually support, based on which source columns are present.
    This is what makes feature engineering dataset-aware instead
    of silently assuming Telco churn column names."""

    columns_set = set(column_names)
    available = []

    for feature_type, requirements in FEATURE_REQUIREMENTS.items():
        required = requirements["required"]
        any_of = requirements["any_of"]

        has_required = required <= columns_set
        has_any_of = (
            not any_of
            or any(column in columns_set for column in any_of)
        )

        if has_required and has_any_of:
            available.append(feature_type)

    return available


def load_and_prepare_dataset(
    dataset_path,
    target_column
):
    df = pd.read_csv(dataset_path)

    if "customerID" in df.columns:
        df = df.drop(
            columns=["customerID"]
        )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df


def create_feature(
    df,
    feature_name,
    feature_type
):
    if feature_type not in FEATURE_REQUIREMENTS:
        raise ValueError(
            f"Unsupported feature type: {feature_type}"
        )

    requirements = FEATURE_REQUIREMENTS[feature_type]
    missing_required = requirements["required"] - set(df.columns)

    if missing_required:
        raise ValueError(
            f"Cannot create '{feature_type}': dataset is missing "
            f"required column(s) {sorted(missing_required)}"
        )

    any_of = requirements["any_of"]

    if any_of and not any(column in df.columns for column in any_of):
        raise ValueError(
            f"Cannot create '{feature_type}': dataset has none of "
            f"the expected column(s) {any_of}"
        )

    if feature_type == "total_charges_per_tenure":
        df[feature_name] = (
            df["TotalCharges"]
            / df["tenure"].replace(0, 1)
        )

    elif feature_type == "monthly_charge_tenure":
        df[feature_name] = (
            df["MonthlyCharges"]
            * df["tenure"]
        )

    elif feature_type == "service_count":
        available_columns = [
            column
            for column in SERVICE_COLUMNS
            if column in df.columns
        ]

        df[feature_name] = (
            df[available_columns]
            .apply(
                lambda row: sum(
                    value == "Yes"
                    for value in row
                ),
                axis=1
            )
        )

    elif feature_type == "support_security_count":
        available_columns = [
            column
            for column in SUPPORT_SECURITY_COLUMNS
            if column in df.columns
        ]

        df[feature_name] = (
            df[available_columns]
            .apply(
                lambda row: sum(
                    value == "Yes"
                    for value in row
                ),
                axis=1
            )
        )

    elif feature_type == "has_streaming":
        available_columns = [
            column
            for column in STREAMING_COLUMNS
            if column in df.columns
        ]

        df[feature_name] = (
            df[available_columns]
            .apply(
                lambda row: int(
                    any(
                        value == "Yes"
                        for value in row
                    )
                ),
                axis=1
            )
        )

    return df


def apply_feature_engineering(
    dataset_path,
    target_column,
    feature_type,
    feature_name
):
    df = load_and_prepare_dataset(
        dataset_path,
        target_column
    )

    df = create_feature(
        df,
        feature_name,
        feature_type
    )

    return df
