from app.state import ResearchState
from app.config import settings
from app.tools.data_tools import split_dataset_indices
from app.logging_config import get_logger

logger = get_logger(__name__)


def split_agent(state: ResearchState) -> dict:
    logger.info("--- SPLIT AGENT ---")

    problem_type = state["research_plan"]["problem_type"]

    splits = split_dataset_indices(
        dataset_path=state["dataset_path"],
        target_column=state["target_column"],
        problem_type=problem_type,
        test_size=settings.TEST_SIZE,
        val_size=settings.VAL_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    logger.info(
        f"Split sizes -> train: {len(splits['train_indices'])}, "
        f"val: {len(splits['val_indices'])}, "
        f"test: {len(splits['test_indices'])} (test is held out "
        "until final_report_agent)"
    )

    return {
        "train_indices": splits["train_indices"],
        "val_indices": splits["val_indices"],
        "test_indices": splits["test_indices"],
        "current_task": "Dataset split into train/validation/test",
    }
