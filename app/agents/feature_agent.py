import pandas as pd

from app.state import ResearchState
from app.logging_config import get_logger

logger = get_logger(__name__)


def feature_agent(state: ResearchState) -> dict:
    logger.info("--- FEATURE AGENT ---")

    dataset_path = state["dataset_path"]
    target_column = state["target_column"]

    df = pd.read_csv(
        dataset_path
    )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    column_names = df.columns.tolist()

    identifier_columns = []

    for column in column_names:
        normalized_column = (
            column.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if normalized_column in {
            "customerid",
            "userid",
            "accountid",
            "recordid",
        }:
            identifier_columns.append(
                column
            )

    numeric_features = [
        column
        for column in numeric_columns
        if column != target_column
        and column not in identifier_columns
    ]

    categorical_features = [
        column
        for column in categorical_columns
        if column != target_column
        and column not in identifier_columns
    ]

    missing_value_columns = [
        column
        for column in df.columns
        if df[column].isnull().any()
    ]

    recommendations = []

    for column in numeric_features:
        recommendations.append(
            {
                "column": column,
                "action": "consider_scaling",
                "reason": (
                    "Numerical feature may benefit "
                    "from scaling depending on the model."
                ),
            }
        )

    for column in categorical_features:
        recommendations.append(
            {
                "column": column,
                "action": "one_hot_encode",
                "reason": (
                    "Categorical feature requires "
                    "encoding before model training."
                ),
            }
        )

    for column in missing_value_columns:
        if column != target_column:
            recommendations.append(
                {
                    "column": column,
                    "action": "impute_missing_values",
                    "reason": (
                        "Missing values should be "
                        "handled before model training."
                    ),
                }
            )

    for column in identifier_columns:
        recommendations.append(
            {
                "column": column,
                "action": "exclude",
                "reason": (
                    "Identifier column should not "
                    "be used as a predictive feature."
                ),
            }
        )

    analysis = {
        "target_column": target_column,
        "identifier_columns": identifier_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_value_columns": missing_value_columns,
        "recommendations": recommendations,
    }

    logger.info(f"Identifier columns: {identifier_columns}")
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Missing-value columns: {missing_value_columns}")
    logger.info(f"Feature analysis complete. Recommendations: {len(recommendations)}")

    return {
        "feature_analysis": analysis,
        "agent_results": {
            **state.get(
                "agent_results",
                {}
            ),
            "feature_agent": analysis,
        },
        "current_task": (
            "Feature analysis completed"
        ),
    }
