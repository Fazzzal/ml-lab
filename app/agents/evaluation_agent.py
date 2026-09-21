from app.state import ResearchState
from app.logging_config import get_logger

logger = get_logger(__name__)


def evaluation_agent(state: ResearchState) -> dict:
    logger.info("--- EVALUATION AGENT ---")

    experiments = state.get(
        "experiments",
        []
    )

    research_plan = state.get(
        "research_plan",
        {}
    )

    problem_type = research_plan.get(
        "problem_type"
    )

    if not experiments:
        logger.info("No experiments available.")

        return {
            "current_task": "No experiments available for evaluation"
        }

    if problem_type == "classification":
        best_experiment = max(
            experiments,
            key=lambda experiment: experiment.get(
                "f1",
                float("-inf")
            )
        )

        metric_name = "f1"
        metric_direction = "higher_is_better"

    else:
        best_experiment = min(
            experiments,
            key=lambda experiment: experiment.get(
                "rmse",
                float("inf")
            )
        )

        metric_name = "rmse"
        metric_direction = "lower_is_better"

    evaluation = {
        "problem_type": problem_type,
        "experiments_evaluated": len(experiments),
        "best_model": best_experiment["model"],
        "best_metric": best_experiment[metric_name],
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        # Note: this metric was computed on the validation split,
        # not test. final_report_agent evaluates the winner on the
        # held-out test set exactly once, separately.
        "metric_basis": "validation",
        "best_experiment": best_experiment,
        "all_results": experiments,
    }

    logger.info(f"Best model: {evaluation['best_model']}")
    logger.info(f"Best {metric_name} (validation): {evaluation['best_metric']}")

    return {
        "evaluation": evaluation,
        "agent_results": {
            **state.get("agent_results", {}),
            "evaluation_agent": evaluation,
        },
        "current_task": (
            "Experiment evaluation completed"
        ),
    }
