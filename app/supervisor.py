from app.state import ResearchState
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def supervisor(
    state: ResearchState
) -> dict:
    logger.info("--- SUPERVISOR ---")

    current_iteration = state.get(
        "retry_count",
        0,
    )

    max_iterations = settings.MAX_ITERATIONS

    if current_iteration >= max_iterations:
        logger.warning(
            "Maximum iteration limit reached. "
            "Stopping the research loop."
        )

        return {
            "next_agent": "finish",
            "retry_count": max_iterations,
            "supervisor_reason": (
                f"Maximum iteration limit "
                f"({max_iterations}) reached; "
                "stopping the research loop."
            ),
            "current_task": (
                "Research stopped: iteration limit reached"
            ),
        }

    retry_count = current_iteration + 1

    critic_analysis = state.get(
        "critic_analysis",
        {},
    )

    decision = critic_analysis.get(
        "decision"
    )

    experiment_type = critic_analysis.get(
        "experiment_type"
    )

    logger.info(
        f"Critic decision: {decision}"
    )

    logger.info(
        f"Experiment type: {experiment_type}"
    )

    logger.info(
        f"Supervisor iteration: "
        f"{retry_count}/{max_iterations}"
    )

    if decision == "feature_engineering":
        logger.info(
            "Routing to feature engineering agent."
        )

        return {
            "next_agent": (
                "feature_engineering_agent"
            ),
            "retry_count": retry_count,
            "supervisor_reason": (
                "Critic requested feature "
                "engineering after baseline "
                "evaluation."
            ),
            "current_task": (
                "Starting feature engineering"
            ),
        }

    if decision == "continue":
        if experiment_type == "baseline_model":
            proposed_experiment = state.get(
                "proposed_experiment",
                {},
            )

            model = proposed_experiment.get(
                "model"
            )

            experiment_history = state.get(
                "experiment_history",
                [],
            )

            experiment_id = (
                f"baseline_model::{model}"
                if model
                else None
            )

            if (
                model
                and experiment_id
                not in experiment_history
            ):
                logger.info(
                    f"Routing to experiment agent: "
                    f"{model}"
                )

                return {
                    "next_agent": (
                        "experiment_agent"
                    ),
                    "retry_count": retry_count,
                    "supervisor_reason": (
                        f"Critic requested baseline "
                        f"experiment: {model}."
                    ),
                    "current_task": (
                        f"Running experiment: {model}"
                    ),
                }

            logger.warning(
                "Critic proposed a baseline model "
                "that has already been run or is invalid."
            )

            return {
                "next_agent": "finish",
                "retry_count": retry_count,
                "supervisor_reason": (
                    "Critic requested a baseline "
                    "experiment that was already "
                    "executed or was invalid."
                ),
                "current_task": (
                    "Duplicate/invalid experiment "
                    "proposal; stopping"
                ),
            }

        logger.warning(
            "Invalid baseline experiment proposal."
        )

        return {
            "next_agent": "finish",
            "retry_count": retry_count,
            "supervisor_reason": (
                "Critic requested continuation "
                "but did not provide a valid "
                "baseline experiment."
            ),
            "current_task": (
                "Invalid experiment proposal"
            ),
        }

    logger.info("Research finished.")

    return {
        "next_agent": "finish",
        "retry_count": retry_count,
        "supervisor_reason": (
            "Critic determined that no further "
            "experiment is currently required."
        ),
        "current_task": (
            "Research completed"
        ),
    }