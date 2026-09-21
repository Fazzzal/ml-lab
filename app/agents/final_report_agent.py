from app.state import ResearchState
from app.tools.ml_tools import run_final_holdout_evaluation
from app.logging_config import get_logger

logger = get_logger(__name__)


def final_report_agent(state: ResearchState) -> dict:
    logger.info("--- FINAL REPORT AGENT ---")

    evaluation = state.get("evaluation", {})
    experiments = state.get("experiments", [])

    if not experiments or not evaluation:
        report = (
            "No experiments were completed during this run, so "
            "no final report could be generated."
        )

        logger.info(report)

        return {
            "final_report": report,
            "current_task": "Final report generated (no experiments)",
        }

    best_experiment = evaluation.get("best_experiment", {})
    problem_type = evaluation.get("problem_type")

    train_indices = state.get("train_indices")
    val_indices = state.get("val_indices")
    test_indices = state.get("test_indices")

    held_out_evaluation = None
    held_out_summary = "Held-out test evaluation was not available."

    if train_indices and val_indices and test_indices:
        try:
            held_out_evaluation = run_final_holdout_evaluation(
                dataset_path=state["dataset_path"],
                target_column=state["target_column"],
                model_name=best_experiment.get("model"),
                problem_type=problem_type,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                parameters=best_experiment.get("applied_params"),
                feature_type=best_experiment.get("feature_type"),
                feature_name=best_experiment.get("feature_name"),
            )

            metric_name = "f1" if problem_type == "classification" else "rmse"

            held_out_summary = (
                f"Held-out test {metric_name}: "
                f"{held_out_evaluation.get(metric_name)} "
                f"(retrained on train+val, evaluated once on a "
                f"test split the critic loop never saw)."
            )

            logger.info(held_out_summary)
        except Exception as error:
            logger.error(f"Held-out evaluation failed: {error}")
            held_out_summary = f"Held-out test evaluation failed: {error}"
    else:
        logger.warning(
            "No train/val/test split found in state; skipping "
            "held-out evaluation."
        )

    lines = [
        "ML Research Lab - Final Report",
        "=" * 40,
        "",
        f"Problem type: {problem_type}",
        f"Experiments run: {len(experiments)}",
        (
            f"Feature engineering attempts: "
            f"{len(state.get('feature_engineering_history', []))}"
        ),
        "",
        f"Best model (by validation {evaluation.get('metric_name')}): "
        f"{evaluation.get('best_model')}",
        f"Validation {evaluation.get('metric_name')}: "
        f"{evaluation.get('best_metric')}",
        "",
        held_out_summary,
        "",
        "Critic feedback trail:",
    ]

    for entry in state.get("critic_feedback", []):
        lines.append(f"  - {entry}")

    lines.append("")
    lines.append(
        f"Stopped because: {state.get('supervisor_reason', 'unknown')}"
    )

    report = "\n".join(lines)

    logger.info("Final report generated.")

    return {
        "held_out_evaluation": held_out_evaluation,
        "final_report": report,
        "current_task": "Final report generated",
    }
