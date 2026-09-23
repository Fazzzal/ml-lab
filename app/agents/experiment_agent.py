from app.state import ResearchState
from app.config import settings
from app.tools.ml_tools import run_experiment, get_supported_models
from app.tools.feature_tools import apply_feature_engineering
from app.logging_config import get_logger


logger = get_logger(__name__)


def _get_validation_metric(
    result: dict,
    problem_type: str,
) -> tuple[str, float | None]:
    if problem_type == "classification":
        metric_name = "f1"
        value = result.get("f1")
    else:
        metric_name = "rmse"
        value = result.get("rmse")

    if value is None:
        return metric_name, None

    return metric_name, float(value)


def _evaluate_experiment(
    result: dict,
    problem_type: str,
    state: ResearchState,
) -> tuple[dict, dict, float | None]:
    metric_name, metric_value = _get_validation_metric(
        result,
        problem_type,
    )

    previous_best = state.get("best_validation_metric")
    previous_best_experiment = state.get("best_experiment", {})

    if metric_value is None:
        result["improvement"] = None
        result["is_best"] = False
        result["metric_name"] = metric_name

        return (
            result,
            previous_best_experiment,
            previous_best,
        )

    if previous_best is None:
        improvement = None
        is_best = True
    elif problem_type == "classification":
        improvement = metric_value - float(previous_best)
        is_best = improvement > 0
    else:
        improvement = float(previous_best) - metric_value
        is_best = improvement > 0

    result["metric_name"] = metric_name
    result["validation_metric"] = metric_value
    result["improvement"] = improvement
    result["is_best"] = is_best

    if is_best:
        best_experiment = dict(result)
        best_validation_metric = metric_value

        logger.info(
            f"New best experiment: "
            f"{metric_name}={metric_value:.6f}"
        )

        return (
            result,
            best_experiment,
            best_validation_metric,
        )

    logger.info(
        f"Experiment did not improve best {metric_name}: "
        f"{metric_value:.6f}"
    )

    return (
        result,
        previous_best_experiment,
        previous_best,
    )


def experiment_agent(state: ResearchState) -> dict:
    logger.info("--- EXPERIMENT AGENT ---")

    proposed_experiment = state.get(
        "proposed_experiment",
        {},
    )

    experiment_type = proposed_experiment.get(
        "experiment_type"
    )

    experiments = list(
        state.get(
            "experiments",
            [],
        )
    )

    experiment_history = list(
        state.get(
            "experiment_history",
            [],
        )
    )

    train_indices = state.get("train_indices")
    val_indices = state.get("val_indices")

    if not train_indices or not val_indices:
        logger.error(
            "No train/val split available. split_agent must run "
            "before any experiment."
        )

        return {
            "current_task": (
                "Cannot run experiment: dataset has not been split"
            )
        }

    problem_type = state[
        "research_plan"
    ][
        "problem_type"
    ]

    if experiment_type == "baseline_model":
        model = proposed_experiment.get(
            "model"
        )

        if not model:
            logger.warning(
                "Baseline experiment has no model."
            )

            return {
                "current_task": (
                    "Invalid baseline experiment"
                )
            }

        experiment_id = (
            f"baseline_model::{model}"
        )

        if experiment_id in experiment_history:
            logger.info(
                f"Experiment already executed: {experiment_id}"
            )

            return {
                "current_task": (
                    "Duplicate experiment prevented"
                )
            }

        supported_models = get_supported_models(
            problem_type
        )

        if model not in supported_models:
            logger.warning(
                f"Unsupported model requested: {model}"
            )

            experiment_history.append(
                experiment_id
            )

            return {
                "experiment_history": (
                    experiment_history
                ),
                "current_task": (
                    f"Rejected unsupported model: {model}"
                ),
            }

        logger.info(
            f"Running proposed baseline: {model}"
        )

        try:
            result = run_experiment(
                dataset_path=state["dataset_path"],
                target_column=state["target_column"],
                model_name=model,
                problem_type=problem_type,
                train_indices=train_indices,
                val_indices=val_indices,
                parameters=proposed_experiment.get(
                    "parameters"
                ),
                cv_folds=settings.CV_FOLDS,
            )
        except Exception as error:
            logger.error(
                f"Experiment failed: {error}"
            )

            experiment_history.append(
                experiment_id
            )

            return {
                "experiment_history": experiment_history,
                "agent_results": {
                    **state.get(
                        "agent_results",
                        {},
                    ),
                    "last_experiment_error": str(error),
                },
                "current_task": (
                    f"Experiment failed: {experiment_id}"
                ),
            }

        result["experiment_type"] = "baseline_model"
        result["experiment_id"] = experiment_id
        result["model"] = model

        (
            result,
            best_experiment,
            best_validation_metric,
        ) = _evaluate_experiment(
            result=result,
            problem_type=problem_type,
            state=state,
        )

        experiments.append(result)
        experiment_history.append(experiment_id)

        logger.info(
            f"Experiment result: {result}"
        )

        return {
            "experiments": experiments,
            "experiment_history": experiment_history,
            "best_experiment": best_experiment,
            "best_validation_metric": best_validation_metric,
            "best_metric_name": result.get(
                "metric_name"
            ),
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "last_experiment": result,
            },
            "current_task": (
                f"Completed experiment: {experiment_id}"
            ),
        }

    if experiment_type == "feature_engineering":
        feature_type = proposed_experiment.get(
            "feature_type"
        )

        feature_name = proposed_experiment.get(
            "feature_name"
        )

        model = proposed_experiment.get(
            "model"
        )

        source_columns = proposed_experiment.get(
            "source_columns",
            [],
        )

        if not feature_type:
            logger.warning(
                "Feature engineering experiment has no "
                "feature type."
            )

            return {
                "current_task": (
                    "Invalid feature engineering proposal"
                )
            }

        if feature_type == "none":
            logger.info(
                "No feature engineering experiment requested."
            )

            return {
                "current_task": (
                    "No feature engineering experiment"
                )
            }

        if not feature_name:
            feature_name = feature_type

        if not model:
            logger.warning(
                "No model specified for feature engineering "
                "experiment."
            )

            return {
                "current_task": (
                    "Feature engineering has no model"
                )
            }

        experiment_id = (
            f"feature_engineering::"
            f"{feature_type}::"
            f"{'::'.join(source_columns)}::"
            f"{model}"
        )

        if experiment_id in experiment_history:
            logger.info(
                f"Feature experiment already executed: "
                f"{experiment_id}"
            )

            return {
                "current_task": (
                    "Duplicate feature experiment prevented"
                )
            }

        supported_models = get_supported_models(
            problem_type
        )

        if model not in supported_models:
            logger.warning(
                f"Unsupported model requested: {model}"
            )

            experiment_history.append(
                experiment_id
            )

            return {
                "experiment_history": (
                    experiment_history
                ),
                "current_task": (
                    f"Rejected unsupported model: {model}"
                ),
            }

        logger.info(
            f"Applying feature engineering: {feature_type}"
        )

        try:
            engineered_df = apply_feature_engineering(
                dataset_path=state["dataset_path"],
                target_column=state["target_column"],
                feature_type=feature_type,
                feature_name=feature_name,
                source_columns=source_columns,
            )

            logger.info(
                f"Running model: {model}"
            )

            result = run_experiment(
                dataset_path=state["dataset_path"],
                target_column=state["target_column"],
                model_name=model,
                problem_type=problem_type,
                train_indices=train_indices,
                val_indices=val_indices,
                parameters=proposed_experiment.get(
                    "parameters"
                ),
                cv_folds=settings.CV_FOLDS,
                dataframe=engineered_df,
            )

        except Exception as error:
            logger.error(
                f"Feature engineering experiment failed: "
                f"{error}"
            )

            experiment_history.append(
                experiment_id
            )

            feature_engineering_history = list(
                state.get(
                    "feature_engineering_history",
                    [],
                )
            )

            feature_engineering_history.append(
                experiment_id
            )

            return {
                "experiment_history": experiment_history,
                "feature_engineering_history": (
                    feature_engineering_history
                ),
                "agent_results": {
                    **state.get(
                        "agent_results",
                        {},
                    ),
                    "last_experiment_error": str(error),
                },
                "current_task": (
                    f"Feature experiment failed: "
                    f"{experiment_id}"
                ),
            }

        result["experiment_type"] = "feature_engineering"
        result["experiment_id"] = experiment_id
        result["model"] = model
        result["feature_type"] = feature_type
        result["feature_name"] = feature_name
        result["source_columns"] = source_columns
        result["note"] = (
            "Feature transformation was created successfully."
        )

        (
            result,
            best_experiment,
            best_validation_metric,
        ) = _evaluate_experiment(
            result=result,
            problem_type=problem_type,
            state=state,
        )

        experiments.append(result)
        experiment_history.append(experiment_id)

        feature_engineering_history = list(
            state.get(
                "feature_engineering_history",
                [],
            )
        )

        feature_engineering_history.append(
            experiment_id
        )

        logger.info(
            f"Feature engineering experiment result: "
            f"{result}"
        )

        return {
            "experiments": experiments,
            "experiment_history": experiment_history,
            "feature_engineering_history": (
                feature_engineering_history
            ),
            "best_experiment": best_experiment,
            "best_validation_metric": best_validation_metric,
            "best_metric_name": result.get(
                "metric_name"
            ),
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "last_experiment": result,
            },
            "current_task": (
                f"Completed feature experiment: "
                f"{experiment_id}"
            ),
        }

    logger.warning(
        f"Unsupported experiment type: {experiment_type}"
    )

    return {
        "current_task": (
            "Unsupported experiment type"
        )
    }