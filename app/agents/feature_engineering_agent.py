from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.state import ResearchState
from app.config import settings
from app.tools.feature_tools import get_available_feature_types
from app.logging_config import get_logger

logger = get_logger(__name__)


llm = ChatGroq(
    model=settings.GROQ_MODEL
)


class FeatureProposal(BaseModel):
    decision: Literal[
        "propose",
        "finish"
    ]

    feature_type: Literal[
        "total_charges_per_tenure",
        "monthly_charge_tenure",
        "service_count",
        "support_security_count",
        "has_streaming",
        "none"
    ]

    feature_name: str | None

    reasoning: str


structured_llm = llm.with_structured_output(
    FeatureProposal
)


FEATURE_ENGINEERING_PROMPT = """
You are the Feature Engineering Agent in an
autonomous machine learning research system.

Your job is to examine the dataset analysis,
current experiment results, and previous feature
engineering attempts.

Decide whether a useful new feature should be
tested.

Based on the columns actually present in THIS dataset, the
following feature transformations are currently supported by the
execution layer (transforms that need columns this dataset
doesn't have have already been excluded from this list):

{available_feature_descriptions}

You may ONLY propose one of the feature types named above. If the
list above is empty, or every listed feature type has already
been tried (see below), return decision = "finish".

Do not propose:
- StandardScaler
- OneHotEncoder
- Missing-value imputation
- Arbitrary Python code
- Unsupported transformations
- Hyperparameter tuning
- A different model

Those operations are handled elsewhere.

Avoid proposing a feature that has already been
tested.

If there is a reasonable untested feature from the list above,
return:

decision = "propose"

If no useful untested feature remains,
return:

decision = "finish"

Dataset analysis:
{feature_analysis}

Research plan:
{research_plan}

Completed experiments:
{experiments}

Previous feature engineering feedback:
{critic_feedback}

Previous feature engineering attempts:
{feature_engineering_history}

The Critic Agent has requested this model:
{requested_model}

If the Critic requested a model, the resulting
feature engineering experiment MUST use that model.

Choose the feature based on the available
evidence. Do not simply choose randomly.

Explain why the feature could provide useful
information for this dataset.
"""

FEATURE_DESCRIPTIONS = {
    "total_charges_per_tenure": (
        "total_charges_per_tenure - Creates TotalCharges / tenure."
    ),
    "monthly_charge_tenure": (
        "monthly_charge_tenure - Creates MonthlyCharges * tenure."
    ),
    "service_count": (
        "service_count - Counts the customer's subscribed services."
    ),
    "support_security_count": (
        "support_security_count - Counts security and technical "
        "support services."
    ),
    "has_streaming": (
        "has_streaming - Indicates whether the customer has at "
        "least one streaming service."
    ),
}


def feature_engineering_agent(
    state: ResearchState
) -> dict:
    logger.info("--- FEATURE ENGINEERING AGENT ---")

    feature_analysis = state.get(
        "feature_analysis",
        {}
    )

    research_plan = state.get(
        "research_plan",
        {}
    )

    experiments = state.get(
        "experiments",
        []
    )

    critic_feedback = state.get(
        "critic_feedback",
        []
    )

    feature_engineering_history = state.get(
        "feature_engineering_history",
        []
    )

    requested_model = state.get(
        "proposed_experiment",
        {}
    ).get(
        "model"
    )

    column_names = state.get(
        "dataset_info", {}
    ).get(
        "column_names", []
    )

    available_feature_types = get_available_feature_types(
        column_names
    )

    if not available_feature_types:
        logger.info(
            "No feature types are supported for this dataset's "
            "columns; skipping feature engineering."
        )

        proposal_dict = {
            "decision": "finish",
            "feature_type": "none",
            "feature_name": None,
            "reasoning": (
                "No supported feature transformation has the "
                "required source columns in this dataset."
            ),
        }

        return {
            "feature_proposal": proposal_dict,
            "proposed_experiment": {
                "experiment_type": "none",
                "feature_type": "none",
                "feature_name": None,
                "model": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get("agent_results", {}),
                "feature_engineering_agent": proposal_dict,
            },
            "current_task": (
                "No feature engineering possible for this dataset"
            ),
        }

    available_feature_descriptions = "\n".join(
        FEATURE_DESCRIPTIONS[feature_type]
        for feature_type in available_feature_types
    )

    prompt = FEATURE_ENGINEERING_PROMPT.format(
        available_feature_descriptions=available_feature_descriptions,
        feature_analysis=feature_analysis,
        research_plan=research_plan,
        experiments=experiments,
        critic_feedback=critic_feedback,
        feature_engineering_history=(
            feature_engineering_history
        ),
        requested_model=requested_model,
    )

    logger.info("Generating feature engineering proposal...")

    proposal = structured_llm.invoke(
        prompt
    )

    # Guard against the LLM re-proposing a feature/model
    # combination that has already been tried, proposing a feature
    # type this dataset can't support (defense in depth beyond the
    # prompt-level restriction above), or proposing "propose" with
    # no usable feature_type/model.
    if proposal.decision == "propose":
        candidate_id = (
            f"feature_engineering::"
            f"{proposal.feature_type}::{requested_model}"
        )

        if (
            proposal.feature_type == "none"
            or proposal.feature_type not in available_feature_types
            or not requested_model
            or candidate_id in feature_engineering_history
        ):
            logger.warning(
                "Feature engineering agent proposed a "
                "duplicate/unsupported/invalid feature; "
                "overriding decision to finish."
            )

            proposal.decision = "finish"
            proposal.feature_type = "none"
            proposal.feature_name = None

    logger.info(f"Decision: {proposal.decision}")
    logger.info(f"Feature: {proposal.feature_type}")
    logger.info(f"Reasoning: {proposal.reasoning}")

    proposal_dict = {
        "decision": proposal.decision,
        "feature_type": proposal.feature_type,
        "feature_name": proposal.feature_name,
        "reasoning": proposal.reasoning,
    }

    if proposal.decision == "finish":
        return {
            "feature_proposal": proposal_dict,
            "proposed_experiment": {
                "experiment_type": "none",
                "feature_type": "none",
                "feature_name": None,
                "model": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {}
                ),
                "feature_engineering_agent": (
                    proposal_dict
                ),
            },
            "current_task": (
                "No further feature engineering available"
            ),
        }

    return {
        "feature_proposal": proposal_dict,
        "proposed_experiment": {
            "experiment_type": (
                "feature_engineering"
            ),
            "feature_type": proposal.feature_type,
            "feature_name": proposal.feature_name,
            "model": requested_model,
            "parameters": {},
        },
        "agent_results": {
            **state.get(
                "agent_results",
                {}
            ),
            "feature_engineering_agent": (
                proposal_dict
            ),
        },
        "current_task": (
            "Feature engineering proposal generated"
        ),
    }
