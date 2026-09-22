import os
from pathlib import Path

# Pin the working directory to wherever this file actually lives,
# regardless of what launched it (PyCharm run configs, a
# different shell cwd, etc. can otherwise leave DATASET_PATH and
# other relative paths in config.py resolving against the wrong
# location, producing a confusing FileNotFoundError even when the
# file and .env are both correct).
os.chdir(Path(__file__).resolve().parent)

from app.graph import build_graph
from app.config import settings
from app.tools.persistence import save_run_snapshot
from app.logging_config import get_logger

logger = get_logger(__name__)


def main():
    graph = build_graph()

    initial_state = {
        "dataset_path": settings.DATASET_PATH,
        "target_column": settings.TARGET_COLUMN,

        "dataset_info": {},
        "feature_analysis": {},
        "research_plan": {},
        "evaluation": {},
        "critic_analysis": {},

        "proposed_experiment": {},

        "agent_results": {},

        "experiments": [],
        "experiment_history": [],
        "feature_engineering_history": [],
        "critic_feedback": [],

        "retry_count": 0,
    }

    invoke_kwargs = {}

    if settings.ENABLE_CHECKPOINTING:
        # Re-running with the same RUN_ID resumes from the last
        # completed node instead of starting over, if a
        # checkpointer is active (see app/graph.py).
        invoke_kwargs["config"] = {
            "configurable": {"thread_id": settings.RUN_ID}
        }

    logger.info("Starting ML Research Lab...")

    try:
        result = graph.invoke(initial_state, **invoke_kwargs)
    except Exception:
        logger.exception(
            "Graph run failed with an unhandled exception. If "
            "ENABLE_CHECKPOINTING=true, re-running with the same "
            "RUN_ID will resume from the last completed step."
        )
        raise

    snapshot_path = save_run_snapshot(result, settings.RUNS_DIR)
    logger.info(f"Run snapshot saved to: {snapshot_path}")

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print("\nDataset Information:")
    print(result.get("dataset_info"))

    print("\nFeature Analysis:")
    print(result.get("feature_analysis"))

    print("\nResearch Plan:")
    print(result.get("research_plan"))

    print("\nExperiments:")
    for experiment in result.get("experiments", []):
        print(experiment)

    print("\nEvaluation (validation metrics):")
    print(result.get("evaluation"))

    print("\nHeld-out Test Evaluation:")
    print(result.get("held_out_evaluation"))

    print("\nCritic Analysis:")
    print(result.get("critic_analysis"))

    print("\nSupervisor reason (final):")
    print(result.get("supervisor_reason"))

    print("\nSupervisor iterations used:")
    print(result.get("retry_count"))

    print("\n==============================")
    print("FINAL REPORT")
    print("==============================")
    print(result.get("final_report"))

    return result


if __name__ == "__main__":
    main()