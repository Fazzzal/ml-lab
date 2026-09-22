from app.state import ResearchState
from app.tools.data_tools import profile_dataset
from app.logging_config import get_logger

logger = get_logger(__name__)


def data_agent(state: ResearchState) -> dict:
    logger.info("--- DATA AGENT ---")

    dataset_path = state["dataset_path"]
    target_column = state["target_column"]

    logger.info(f"Profiling dataset: {dataset_path}")

    profile = profile_dataset(dataset_path)

    # Fail fast and clearly here, rather than several nodes later
    # as a cryptic pandas KeyError deep inside sklearn splitting
    # logic. A mismatched TARGET_COLUMN (e.g. leftover config from
    # a previous dataset) is a common, easy mistake to make.
    if target_column not in profile["column_names"]:
        raise ValueError(
            f"Target column '{target_column}' was not found in "
            f"'{dataset_path}'. Available columns: "
            f"{profile['column_names']}. Check TARGET_COLUMN in "
            f"your .env matches this dataset."
        )

    result = {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "numeric_columns": profile["numeric_columns"],
        "categorical_columns": profile["categorical_columns"],
        "missing_values": profile["missing_values"],
        "duplicates": profile["duplicates"],
        "dtypes": profile["dtypes"],
    }

    logger.info("Dataset profiling complete.")

    return {
        "dataset_info": result,
        "agent_results": {
            **state.get("agent_results", {}),
            "data_agent": result,
        },
        "current_task": "Dataset profiling completed",
    }