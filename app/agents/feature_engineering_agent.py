from app.logging_config import get_logger
from app.tools.feature_tools import (
    TRANSFORMATIONS,
    get_feature_candidates,
    get_column_types,
    load_and_prepare_dataset,
)

logger = get_logger(__name__)

MAX_FEATURE_EXPERIMENTS = 3


def _get_attempted_features(state):
    attempted = set()

    histories = []

    histories.extend(
        state.get("feature_engineering_history", [])
    )

    for experiment in state.get("experiments", []):
        if experiment.get("experiment_type") == "feature_engineering":
            experiment_id = experiment.get("experiment_id")

            if experiment_id:
                histories.append(experiment_id)

    for experiment_id in histories:
        if not experiment_id:
            continue

        parts = experiment_id.split("::")

        if len(parts) < 3:
            continue

        if parts[0] != "feature_engineering":
            continue

        feature_type = parts[1]

        if len(parts) >= 5:
            source_columns = tuple(parts[2:-1])
            model = parts[-1]
        else:
            source_columns = tuple(parts[2:])
            model = None

        attempted.add(
            (
                feature_type,
                source_columns,
                model,
            )
        )

    return attempted


def _get_attempted_feature_transformations(state):
    attempted = set()

    for experiment_id in state.get(
        "feature_engineering_history",
        [],
    ):
        if not experiment_id:
            continue

        parts = experiment_id.split("::")

        if len(parts) < 4:
            continue

        if parts[0] != "feature_engineering":
            continue

        feature_type = parts[1]

        if len(parts) >= 5:
            source_columns = tuple(parts[2:-1])
        else:
            source_columns = tuple(parts[2:])

        attempted.add(
            (
                feature_type,
                source_columns,
            )
        )

    for experiment in state.get("experiments", []):
        if experiment.get("experiment_type") != "feature_engineering":
            continue

        feature_type = experiment.get("feature_type")
        source_columns = experiment.get(
            "source_columns",
            [],
        )

        if feature_type and source_columns:
            attempted.add(
                (
                    feature_type,
                    tuple(source_columns),
                )
            )

    return attempted


def _get_feature_name(feature_type, source_columns):
    return (
        f"{feature_type}__"
        + "__".join(source_columns)
    )


def _get_requested_model(state):
    critic_analysis = state.get(
        "critic_analysis",
        {},
    )

    for key in (
        "model",
        "target_model",
        "requested_model",
        "experiment_model",
    ):
        model = critic_analysis.get(key)

        if model:
            return model

    proposed_experiment = state.get(
        "proposed_experiment",
        {},
    )

    model = proposed_experiment.get("model")

    if model:
        return model

    return (
        state.get(
            "evaluation",
            {},
        ).get("best_model")
        or "Logistic Regression"
    )


def _select_next_feature(
    df,
    target_column,
    state,
):
    attempted = _get_attempted_feature_transformations(
        state
    )

    for feature_type in TRANSFORMATIONS:
        candidates = get_feature_candidates(
            dataframe=df,
            feature_type=feature_type,
            target_column=target_column,
        )

        for source_columns in candidates:
            source_columns_tuple = tuple(
                source_columns
            )

            if (
                feature_type,
                source_columns_tuple,
            ) in attempted:
                continue

            feature_name = _get_feature_name(
                feature_type,
                source_columns,
            )

            return {
                "feature_type": feature_type,
                "source_columns": list(source_columns),
                "feature_name": feature_name,
                "reason": (
                    "Selected the next supported feature "
                    "transformation that has not yet "
                    "been tested."
                ),
            }

    return None


def feature_engineering_agent(state):
    logger.info(
        "--- FEATURE ENGINEERING AGENT ---"
    )

    dataset_path = state["dataset_path"]
    target_column = state["target_column"]

    df = load_and_prepare_dataset(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    column_types = get_column_types(
        df,
        target_column=target_column,
    )

    logger.info(
        f"Numeric columns: {column_types['numeric']}"
    )

    logger.info(
        f"Categorical columns: "
        f"{column_types['categorical']}"
    )

    attempted_features = (
        _get_attempted_feature_transformations(
            state
        )
    )

    logger.info(
        f"Unique feature transformations already "
        f"tested: {len(attempted_features)}"
    )

    if len(attempted_features) >= MAX_FEATURE_EXPERIMENTS:
        logger.info(
            "Maximum feature engineering experiments reached."
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "Maximum feature engineering "
                    "experiment budget reached."
                ),
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": (
                        "Maximum feature engineering "
                        "experiment budget reached."
                    ),
                },
            },
            "current_task": (
                "Feature engineering budget exhausted"
            ),
        }

    selected_feature = _select_next_feature(
        df=df,
        target_column=target_column,
        state=state,
    )

    if selected_feature is None:
        logger.info(
            "No untested valid feature transformations remain."
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "No untested valid feature "
                    "transformations remain."
                ),
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": (
                        "No untested valid feature "
                        "transformations remain."
                    ),
                },
            },
            "current_task": (
                "No new feature transformation available"
            ),
        }

    feature_type = selected_feature[
        "feature_type"
    ]

    source_columns = selected_feature[
        "source_columns"
    ]

    feature_name = selected_feature[
        "feature_name"
    ]

    reason = selected_feature["reason"]

    requested_model = _get_requested_model(
        state
    )

    proposed_experiment = {
        "experiment_type": "feature_engineering",
        "feature_type": feature_type,
        "feature_name": feature_name,
        "source_columns": source_columns,
        "model": requested_model,
        "reason": reason,
    }

    logger.info(
        f"Selected model: {requested_model}"
    )

    logger.info(
        f"Feature type: {feature_type}"
    )

    logger.info(
        f"Source columns: {source_columns}"
    )

    logger.info(
        f"Feature name: {feature_name}"
    )

    logger.info(
        "Feature selection completed without an LLM call."
    )

    return {
        "feature_proposal": {
            "decision": "propose",
            "feature_type": feature_type,
            "feature_name": feature_name,
            "source_columns": source_columns,
            "reason": reason,
        },
        "proposed_experiment": proposed_experiment,
        "agent_results": {
            **state.get(
                "agent_results",
                {},
            ),
            "feature_engineering_agent": {
                "status": "complete",
                "feature_type": feature_type,
                "source_columns": source_columns,
                "feature_name": feature_name,
                "model": requested_model,
                "reason": reason,
            },
        },
        "current_task": (
            f"Selected {requested_model} with "
            f"{feature_type}"
        ),
    }