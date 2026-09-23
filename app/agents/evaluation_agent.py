from app.state import ResearchState
from app.logging_config import get_logger

logger = get_logger(__name__)


def _get_metric_name(state):
    problem_type = state.get(
        "research_plan",
        {},
    ).get("problem_type")

    if problem_type == "regression":
        return "rmse"

    return "f1"


def _get_metric_direction(metric_name):
    if metric_name == "rmse":
        return "lower_is_better"

    return "higher_is_better"


def evaluation_agent(
    state: ResearchState,
) -> dict:
    logger.info("--- EVALUATION AGENT ---")

    experiments = list(
        state.get(
            "experiments",
            [],
        )
    )

    problem_type = state.get(
        "research_plan",
        {},
    ).get(
        "problem_type",
        "classification",
    )

    if not experiments:
        logger.warning(
            "No experiments available for evaluation."
        )

        evaluation = {
            "problem_type": problem_type,
            "experiments_evaluated": 0,
            "best_model": None,
            "best_metric": None,
            "metric_name": (
                "rmse"
                if problem_type == "regression"
                else "f1"
            ),
            "metric_direction": (
                "lower_is_better"
                if problem_type == "regression"
                else "higher_is_better"
            ),
            "metric_basis": "validation",
            "best_experiment": None,
            "all_results": [],
            "held_out_test": None,
        }

        return {
            "evaluation": evaluation,
            "best_experiment": {},
            "best_validation_metric": 0.0,
            "best_metric_name": evaluation[
                "metric_name"
            ],
            "current_task": (
                "No experiments available for evaluation"
            ),
        }

    metric_name = _get_metric_name(state)

    metric_direction = _get_metric_direction(
        metric_name
    )

    valid_experiments = []

    for experiment in experiments:
        if experiment.get(
            "validation_metric"
        ) is None:
            continue

        valid_experiments.append(
            experiment
        )

    if not valid_experiments:
        logger.warning(
            "Experiments exist, but none contain "
            "a validation metric."
        )

        evaluation = {
            "problem_type": problem_type,
            "experiments_evaluated": len(
                experiments
            ),
            "best_model": None,
            "best_metric": None,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "metric_basis": "validation",
            "best_experiment": None,
            "all_results": experiments,
            "held_out_test": state.get(
                "evaluation",
                {},
            ).get(
                "held_out_test"
            ),
        }

        return {
            "evaluation": evaluation,
            "current_task": (
                "Evaluation completed without "
                "a valid validation metric"
            ),
        }

    if metric_direction == "higher_is_better":
        best_experiment = max(
            valid_experiments,
            key=lambda experiment: experiment.get(
                "validation_metric",
                float("-inf"),
            ),
        )
    else:
        best_experiment = min(
            valid_experiments,
            key=lambda experiment: experiment.get(
                "validation_metric",
                float("inf"),
            ),
        )

    best_model = best_experiment.get(
        "model"
    )

    best_metric = best_experiment.get(
        "validation_metric"
    )

    logger.info(
        f"Best model: {best_model}"
    )

    logger.info(
        f"Best {metric_name} "
        f"(validation): {best_metric}"
    )

    existing_evaluation = state.get(
        "evaluation",
        {},
    )

    held_out_test = existing_evaluation.get(
        "held_out_test"
    )

    evaluation = {
        "problem_type": problem_type,
        "experiments_evaluated": len(
            valid_experiments
        ),
        "best_model": best_model,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        "metric_basis": "validation",
        "best_experiment": best_experiment,
        "all_results": experiments,
        "held_out_test": held_out_test,
    }

    return {
        "evaluation": evaluation,
        "best_experiment": best_experiment,
        "best_validation_metric": best_metric,
        "best_metric_name": metric_name,
        "current_task": (
            "Validation evaluation completed"
        ),
    }