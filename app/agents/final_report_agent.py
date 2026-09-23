from app.state import ResearchState
from app.logging_config import get_logger
from app.tools.ml_tools import run_final_holdout_evaluation


logger = get_logger(__name__)


def _build_report(
    state,
    evaluation,
    critic_analysis,
):
    research_plan = state.get(
        "research_plan",
        {}
    )

    experiments = state.get(
        "experiments",
        []
    )

    feature_history = state.get(
        "feature_engineering_history",
        []
    )

    problem_type = research_plan.get(
        "problem_type",
        "unknown"
    )

    metric_name = evaluation.get(
        "metric_name",
        "metric"
    )

    best_model = evaluation.get(
        "best_model"
    )

    best_metric = evaluation.get(
        "best_metric"
    )

    held_out_test = evaluation.get(
        "held_out_test"
    )

    lines = []

    lines.append(
        "ML Research Lab - Final Report"
    )

    lines.append(
        "=" * 40
    )

    lines.append("")

    lines.append(
        f"Problem type: {problem_type}"
    )

    lines.append(
        f"Experiments run: {len(experiments)}"
    )

    lines.append(
        "Feature engineering attempts: "
        f"{len(feature_history)}"
    )

    lines.append("")

    lines.append(
        f"Best model (by validation "
        f"{metric_name}): {best_model}"
    )

    lines.append(
        f"Validation {metric_name}: "
        f"{best_metric}"
    )

    lines.append("")

    if held_out_test:
        if problem_type == "classification":
            test_f1 = held_out_test.get("f1")
            test_accuracy = held_out_test.get(
                "accuracy"
            )

            lines.append(
                f"Held-out test f1: {test_f1}"
            )

            lines.append(
                f"Held-out test accuracy: "
                f"{test_accuracy}"
            )

        else:
            test_rmse = held_out_test.get(
                "rmse"
            )

            test_r2 = held_out_test.get(
                "r2"
            )

            lines.append(
                f"Held-out test rmse: "
                f"{test_rmse}"
            )

            lines.append(
                f"Held-out test r2: "
                f"{test_r2}"
            )

        lines.append("")

        lines.append(
            "The final model was retrained "
            "on train + validation data and "
            "evaluated once on the held-out "
            "test split."
        )

    else:
        lines.append(
            "Held-out test evaluation: "
            "not available."
        )

    lines.append("")

    lines.append(
        "Critic feedback trail:"
    )

    feedback_trail = state.get(
        "critic_feedback",
        []
    )

    if feedback_trail:
        for feedback in feedback_trail:
            lines.append(
                f"  - {feedback}"
            )
    elif critic_analysis:
        recommendation = critic_analysis.get(
            "recommendation",
            ""
        )

        if recommendation:
            lines.append(
                f"  - {recommendation}"
            )
        else:
            lines.append(
                "  - No critic feedback recorded."
            )
    else:
        lines.append(
            "  - No critic feedback recorded."
        )

    lines.append("")

    lines.append(
        "Stopped because: "
        + state.get(
            "supervisor_reason",
            "Research completed."
        )
    )

    return "\n".join(lines)


def final_report_agent(
    state: ResearchState,
) -> dict:
    logger.info(
        "--- FINAL REPORT AGENT ---"
    )

    existing_evaluation = dict(
        state.get(
            "evaluation",
            {}
        )
    )

    best_experiment = (
        state.get(
            "best_experiment"
        )
        or existing_evaluation.get(
            "best_experiment"
        )
    )

    if not best_experiment:
        logger.warning(
            "No best experiment available."
        )

        final_report = (
            "ML Research Lab - Final Report\n\n"
            "No completed experiment was "
            "available for final evaluation."
        )

        return {
            "final_report": final_report,
            "current_task": (
                "Final report generated"
            ),
        }

    dataset_path = state.get(
        "dataset_path"
    )

    target_column = state.get(
        "target_column"
    )

    problem_type = state.get(
        "research_plan",
        {}
    ).get(
        "problem_type",
        "classification"
    )

    train_indices = state.get(
        "train_indices",
        []
    )

    val_indices = state.get(
        "val_indices",
        []
    )

    test_indices = state.get(
        "test_indices",
        []
    )

    model_name = best_experiment.get(
        "model"
    )

    parameters = best_experiment.get(
        "parameters",
        best_experiment.get(
            "applied_params",
            {}
        )
    )

    feature_type = best_experiment.get(
        "feature_type"
    )

    feature_name = best_experiment.get(
        "feature_name"
    )

    logger.info(
        f"Final model: {model_name}"
    )

    logger.info(
        "Running final held-out evaluation "
        "on train + validation versus test."
    )

    try:
        held_out_test = run_final_holdout_evaluation(
            dataset_path=dataset_path,
            target_column=target_column,
            model_name=model_name,
            problem_type=problem_type,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters=parameters,
            feature_type=feature_type,
            feature_name=feature_name,
        )

    except Exception as error:
        logger.exception(
            "Final held-out evaluation failed."
        )

        existing_evaluation[
            "held_out_test"
        ] = None

        existing_evaluation[
            "held_out_test_completed"
        ] = False

        final_report = _build_report(
            state=state,
            evaluation=existing_evaluation,
            critic_analysis=state.get(
                "critic_analysis",
                {}
            ),
        )

        final_report += (
            "\n\nFinal holdout evaluation error: "
            f"{error}"
        )

        return {
            "evaluation": existing_evaluation,
            "final_report": final_report,
            "current_task": (
                "Final report generated "
                "with holdout evaluation error"
            ),
        }

    existing_evaluation[
        "held_out_test"
    ] = held_out_test

    existing_evaluation[
        "held_out_test_completed"
    ] = True

    existing_evaluation[
        "held_out_test_basis"
    ] = (
        "retrained_on_train_plus_validation"
    )

    final_report = _build_report(
        state=state,
        evaluation=existing_evaluation,
        critic_analysis=state.get(
            "critic_analysis",
            {}
        ),
    )

    logger.info(
        "Final report generated."
    )

    return {
        "evaluation": existing_evaluation,
        "final_report": final_report,
        "current_task": (
            "Final report generated"
        ),
    }