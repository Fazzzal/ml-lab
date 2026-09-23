from app.state import ResearchState
from app.logging_config import get_logger
from app.tools.feature_tools import (
    TRANSFORMATIONS,
    get_feature_candidates,
    get_column_types,
    load_and_prepare_dataset,
)


logger = get_logger(__name__)


MAX_FEATURE_EXPERIMENTS = 3


def _get_attempted_feature_transformations(
    state: ResearchState,
) -> set[tuple[str, tuple[str, ...]]]:
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

    for experiment in state.get(
        "experiments",
        [],
    ):
        if experiment.get(
            "experiment_type"
        ) != "feature_engineering":
            continue

        feature_type = experiment.get(
            "feature_type"
        )

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


def _get_requested_model(
    state: ResearchState,
) -> str | None:
    critic_analysis = state.get(
        "critic_analysis",
        {},
    )

    model = critic_analysis.get(
        "model"
    )

    if model:
        return model

    proposed_experiment = state.get(
        "proposed_experiment",
        {},
    )

    model = proposed_experiment.get(
        "model"
    )

    if model:
        return model

    return None


def _get_requested_feature_type(
    state: ResearchState,
) -> str | None:
    critic_analysis = state.get(
        "critic_analysis",
        {},
    )

    feature_type = critic_analysis.get(
        "feature_type"
    )

    if feature_type:
        return feature_type

    proposed_experiment = state.get(
        "proposed_experiment",
        {},
    )

    feature_type = proposed_experiment.get(
        "feature_type"
    )

    if feature_type:
        return feature_type

    return None


def _get_feature_name(
    feature_type: str,
    source_columns: list[str],
) -> str:
    return (
        f"{feature_type}__"
        + "__".join(source_columns)
    )


def _select_next_feature(
    df,
    target_column: str,
    feature_type: str,
    state: ResearchState,
):
    attempted = _get_attempted_feature_transformations(
        state
    )

    if feature_type not in TRANSFORMATIONS:
        logger.warning(
            f"Unsupported requested feature type: "
            f"{feature_type}"
        )
        return None

    candidates = get_feature_candidates(
        dataframe=df,
        feature_type=feature_type,
        target_column=target_column,
    )

    logger.info(
        f"Requested feature type: {feature_type}"
    )

    logger.info(
        f"Valid candidates for {feature_type}: "
        f"{len(candidates)}"
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
            list(source_columns),
        )

        return {
            "feature_type": feature_type,
            "source_columns": list(
                source_columns
            ),
            "feature_name": feature_name,
            "reason": (
                "Selected the next valid and "
                "untested feature transformation "
                "requested by the Critic."
            ),
        }

    return None


def feature_engineering_agent(
    state: ResearchState,
) -> dict:
    logger.info(
        "--- FEATURE ENGINEERING AGENT ---"
    )

    dataset_path = state[
        "dataset_path"
    ]

    target_column = state[
        "target_column"
    ]

    df = load_and_prepare_dataset(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    column_types = get_column_types(
        df,
        target_column=target_column,
    )

    logger.info(
        f"Numeric columns: "
        f"{column_types['numeric']}"
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
        f"Unique feature transformations "
        f"already tested: "
        f"{len(attempted_features)}"
    )

    if len(attempted_features) >= MAX_FEATURE_EXPERIMENTS:
        logger.info(
            "Maximum feature engineering "
            "experiments reached."
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
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "feature_type": None,
                "parameters": {},
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

    requested_model = _get_requested_model(
        state
    )

    requested_feature_type = (
        _get_requested_feature_type(
            state
        )
    )

    logger.info(
        f"Critic requested model: "
        f"{requested_model}"
    )

    logger.info(
        f"Critic requested feature type: "
        f"{requested_feature_type}"
    )

    if not requested_model:
        logger.warning(
            "Feature engineering was requested "
            "without a model. Refusing to silently "
            "select a model."
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "Critic did not specify a model "
                    "for the feature engineering "
                    "experiment."
                ),
            },
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "feature_type": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": (
                        "Critic did not specify a "
                        "model."
                    ),
                },
            },
            "current_task": (
                "Feature engineering stopped because "
                "Critic did not specify a model"
            ),
        }

    if not requested_feature_type:
        logger.warning(
            "Feature engineering was requested "
            "without a feature type."
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "Critic did not specify a feature "
                    "transformation."
                ),
            },
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "feature_type": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": (
                        "Critic did not specify a "
                        "feature transformation."
                    ),
                },
            },
            "current_task": (
                "Feature engineering stopped because "
                "Critic did not specify a feature type"
            ),
        }

    if requested_feature_type not in TRANSFORMATIONS:
        logger.warning(
            f"Unsupported requested feature type: "
            f"{requested_feature_type}"
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "Critic requested an unsupported "
                    "feature transformation."
                ),
            },
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "feature_type": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": (
                        "Unsupported feature "
                        "transformation requested."
                    ),
                },
            },
            "current_task": (
                "Unsupported feature transformation"
            ),
        }

    selected_feature = _select_next_feature(
        df=df,
        target_column=target_column,
        feature_type=requested_feature_type,
        state=state,
    )

    if selected_feature is None:
        logger.info(
            "No untested valid feature candidates "
            "remain for the Critic's requested "
            "transformation."
        )

        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": requested_feature_type,
                "feature_name": "",
                "source_columns": [],
                "reason": (
                    "No untested valid source-column "
                    "combination remains for the "
                    "requested transformation."
                ),
            },
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "feature_type": None,
                "parameters": {},
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
                        "candidate remains."
                    ),
                    "requested_model": requested_model,
                    "requested_feature_type": (
                        requested_feature_type
                    ),
                },
            },
            "current_task": (
                "No new feature candidate available"
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

    reason = selected_feature[
        "reason"
    ]

    proposed_experiment = {
        "experiment_type": (
            "feature_engineering"
        ),
        "feature_type": feature_type,
        "feature_name": feature_name,
        "source_columns": source_columns,
        "model": requested_model,
        "parameters": (
            state.get(
                "critic_analysis",
                {},
            ).get(
                "parameters",
                {},
            )
        ),
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
        "Feature selection followed the "
        "Critic's requested transformation."
    )

    return {
        "feature_proposal": {
            "decision": "propose",
            "feature_type": feature_type,
            "feature_name": feature_name,
            "source_columns": source_columns,
            "reason": reason,
        },
        "proposed_experiment": (
            proposed_experiment
        ),
        "agent_results": {
            **state.get(
                "agent_results",
                {},
            ),
            "feature_engineering_agent": {
                "status": "complete",
                "model": requested_model,
                "feature_type": feature_type,
                "source_columns": source_columns,
                "feature_name": feature_name,
                "reason": reason,
            },
        },
        "current_task": (
            "Feature engineering experiment proposed"
        ),
    }