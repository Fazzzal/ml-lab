from typing import Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from app.state import ResearchState
from app.config import settings
from app.tools.ml_tools import get_supported_models
from app.logging_config import get_logger

logger = get_logger(__name__)


class ResearchPlan(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(
        description="The type of supervised learning problem."
    )

    models: list[str] = Field(
        description="Recommended machine learning models to evaluate."
    )

    reasoning: str = Field(
        description="Reasoning behind the proposed experimental plan."
    )


llm = ChatGroq(
    model=settings.GROQ_MODEL
)

structured_llm = llm.with_structured_output(
    ResearchPlan
)

RESEARCH_PROMPT = """
You are the research scientist in an autonomous machine learning research lab.

Your job is to analyze the dataset and feature information and create an
initial machine learning experiment plan.

Dataset information:
{dataset_info}

Feature analysis:
{feature_analysis}

Target column:
{target_column}

Currently supported models and execution capabilities:

Classification:
{classification_models}

Regression:
{regression_models}

The execution layer currently performs:

- Train/validation/test splitting (stratified for classification)
- Missing-value imputation
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Model training
- Cross-validation on the training split
- Prediction
- Accuracy and weighted F1 for classification
- RMSE and R2 for regression

The execution layer does NOT currently perform:

- Grid search / random search hyperparameter tuning
- ROC-AUC calculation
- Precision/recall calculation
- Automated feature selection

Determine whether this is a classification or regression problem.

Choose suitable models only from the supported model list above.

The initial plan should establish a diverse baseline by selecting models
with different modeling approaches (a linear baseline, a tree-based
bagging approach, and a tree-based boosting approach).

Important rules:

1. Do not choose models that are not in the supported model list above.

2. Do not claim that grid search, random search, ROC-AUC, precision,
   recall, or automated feature selection will be performed, because
   these capabilities are not currently implemented.

3. Do not refer to sklearn GradientBoostingClassifier or
   GradientBoostingRegressor as XGBoost.

4. Do not treat the target column as an input feature.

5. Keep the initial research plan focused on establishing baseline
   performance using the supported models.

6. Your reasoning must be based only on the supplied dataset information,
   feature analysis, and capabilities of the execution layer.

The goal is to establish a reliable baseline that can later be expanded
through additional experiments proposed by the critic agent.
"""


def research_agent(state: ResearchState) -> dict:
    logger.info("--- RESEARCH AGENT ---")
    logger.info("Generating research plan...")

    dataset_info = state.get(
        "dataset_info",
        {}
    )

    feature_analysis = state.get(
        "feature_analysis",
        {}
    )

    target_column = state["target_column"]

    prompt = RESEARCH_PROMPT.format(
        dataset_info=dataset_info,
        feature_analysis=feature_analysis,
        target_column=target_column,
        classification_models=get_supported_models("classification"),
        regression_models=get_supported_models("regression"),
    )

    research_plan = structured_llm.invoke(
        prompt
    )

    # The LLM's `models` field is still free text under the hood
    # (structured output constrains the shape, not the values), so
    # filter to what the execution layer can actually run and fall
    # back to the full supported list if nothing survives.
    supported_models = get_supported_models(
        research_plan.problem_type
    )

    valid_models = [
        model
        for model in research_plan.models
        if model in supported_models
    ]

    if not valid_models:
        logger.warning(
            f"Research agent proposed no supported models "
            f"({research_plan.models}); falling back to the full "
            f"supported list for {research_plan.problem_type}."
        )

        valid_models = supported_models

    research_plan.models = valid_models

    logger.info(f"Problem type: {research_plan.problem_type}")
    logger.info(f"Models: {research_plan.models}")
    logger.info(f"Reasoning: {research_plan.reasoning}")

    first_model = research_plan.models[0]

    proposed_experiment = {
        "experiment_type": "baseline_model",
        "model": first_model,
        "parameters": {},
        "reason": (
            "Initial baseline experiment selected "
            "from the research plan."
        ),
    }

    return {
        "research_plan": research_plan.model_dump(),

        "proposed_experiment": proposed_experiment,

        "agent_results": {
            **state.get(
                "agent_results",
                {}
            ),
            "research_agent": research_plan.model_dump(),
        },

        "current_task": (
            "Research plan generated and initial "
            "baseline experiment proposed"
        ),
    }
