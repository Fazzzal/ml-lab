from typing import Literal

from pydantic import BaseModel, Field

from app.logging_config import get_logger
from app.config import settings
from app.tools.feature_tools import (
    TRANSFORMATIONS,
    get_feature_candidates,
    get_column_types,
    load_and_prepare_dataset,
)

from langchain_groq import ChatGroq

logger = get_logger(__name__)


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=settings.LLM_MAX_RETRIES,
)


class FeatureProposal(BaseModel):
    action: Literal["propose", "stop"] = Field(
        description="Whether to propose a feature transformation or stop."
    )
    feature_type: str = Field(
        description="Feature transformation type."
    )
    source_columns: list[str] = Field(
        default_factory=list,
        description="Existing columns used to create the feature."
    )
    feature_name: str = Field(
        default="",
        description="Name of the engineered feature."
    )
    reason: str = Field(
        default="",
        description="Why this feature may improve validation performance."
    )


def feature_engineering_agent(state):
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]

    df = load_and_prepare_dataset(
        dataset_path=dataset_path,
        target_column=target_column,
    )

    column_types = get_column_types(
        df,
        target_column=target_column,
    )

    available_types = list(TRANSFORMATIONS.keys())

    candidate_summary = {}

    for feature_type in available_types:
        candidates = get_feature_candidates(
            dataframe=df,
            feature_type=feature_type,
            target_column=target_column,
        )

        if candidates:
            candidate_summary[feature_type] = candidates[:30]

    prompt = f"""
You are the feature engineering agent in an autonomous ML research system.

Your job is to propose ONE useful feature engineering experiment.

Dataset target:
{target_column}

Numeric columns:
{column_types["numeric"]}

Categorical columns:
{column_types["categorical"]}

Available transformations:
{available_types}

Valid candidate source columns for each transformation:
{candidate_summary}

Existing feature engineering history:
{state.get("feature_engineering_history", [])}

Previous critic feedback:
{state.get("critic_feedback", [])}

Rules:

1. Only use transformations listed under Available transformations.
2. Only use source columns from the provided candidate lists.
3. Never use the target column.
4. Do not invent columns.
5. Propose exactly one feature.
6. Prefer a transformation that addresses the critic feedback.
7. Avoid repeating an already attempted transformation with the same source columns.
8. If there is no sensible new feature to test, choose stop.
"""

    structured_llm = llm.with_structured_output(
        FeatureProposal
    )

    proposal = structured_llm.invoke(prompt)

    feature_type = proposal.feature_type
    source_columns = proposal.source_columns

    if proposal.action == "propose":
        if feature_type not in TRANSFORMATIONS:
            proposal.action = "stop"

        else:
            valid_candidates = get_feature_candidates(
                dataframe=df,
                feature_type=feature_type,
                target_column=target_column,
            )

            if source_columns not in valid_candidates:
                proposal.action = "stop"

    if proposal.action == "stop":
        return {
            "feature_proposal": {
                "decision": "stop",
                "feature_type": "",
                "feature_name": "",
                "source_columns": [],
                "reason": proposal.reason,
            },
            "agent_results": {
                **state.get("agent_results", {}),
                "feature_engineering_agent": {
                    "status": "stopped",
                    "reason": proposal.reason,
                },
            }
        }

    feature_name = proposal.feature_name.strip()

    if not feature_name:
        feature_name = (
            f"{feature_type}__"
            + "__".join(source_columns)
        )

    best_model = state.get("evaluation", {}).get("best_model")

    if not best_model:
        best_model = "Logistic Regression"

    proposed_experiment = {
        "experiment_type": "feature_engineering",
        "feature_type": feature_type,
        "feature_name": feature_name,
        "source_columns": source_columns,
        "model": best_model,
        "reason": proposal.reason,
    }

    logger.info("--- FEATURE ENGINEERING AGENT ---")
    logger.info(f"Selected model: {best_model}")
    logger.info(f"Feature type: {feature_type}")
    logger.info(f"Source columns: {source_columns}")
    logger.info(f"Feature name: {feature_name}")

    history_entry = (
        f"{feature_type}::"
        f"{'::'.join(source_columns)}"
    )



    return {
        "feature_proposal": {
            "decision": "propose",
            "feature_type": feature_type,
            "feature_name": feature_name,
            "source_columns": source_columns,
            "reason": proposal.reason,
        },
        "proposed_experiment": proposed_experiment,

        "agent_results": {
            **state.get("agent_results", {}),
            "feature_engineering_agent": {
                "status": "complete",
                "feature_type": feature_type,
                "source_columns": source_columns,
                "feature_name": feature_name,
                "reason": proposal.reason,
            },
        },
    }