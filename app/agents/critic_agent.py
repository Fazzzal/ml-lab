
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.state import ResearchState
from app.config import settings
from app.tools.ml_tools import get_supported_models, ALLOWED_PARAMETERS
from app.tools.summarize import summarize_experiments, recent_feedback
from app.logging_config import get_logger


logger = get_logger(__name__)

MAX_FEATURE_EXPERIMENTS = 3


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=settings.LLM_MAX_RETRIES,
)


class CriticDecision(BaseModel):
    decision: Literal[
        "continue",
        "feature_engineering",
        "finish",
    ]

    experiment_type: Literal[
        "baseline_model",
        "feature_engineering",
        "none",
    ]

    model: str | None

    parameters: dict

    concerns: list[str]

    recommendation: str

    reasoning: str


structured_llm = llm.with_structured_output(
    CriticDecision
)


CRITIC_PROMPT = """
You are the Critic Agent in an autonomous machine learning research system.

Your job is to examine the experiments completed so far and decide what should happen next.

The baseline models in the research plan are:

{research_plan}

Completed experiments:

{experiments}

Feature engineering attempts already made:

{feature_engineering_history}

Recent feedback from your previous decisions:

{critic_feedback}

Allowed hyperparameters per model:

{allowed_parameters}

You have three possible decisions.

1. continue

Use this only when another baseline model from the research plan still needs to be evaluated.

2. feature_engineering

Use this only when:
- all baseline models have been evaluated
- fewer than the allowed number of feature engineering experiments have been attempted
- there is a reasonable opportunity to improve the current best validation result

3. finish

Use this when:
- all baseline models have been evaluated
- the feature engineering budget has been exhausted
- there is no useful remaining experiment
- the latest feature engineering experiment did not provide a meaningful improvement
- or the available experiments are otherwise exhausted

Important rules:

- Do not propose unsupported models.
- Only propose hyperparameters listed in the allowed set.
- Do not propose scaling.
- Do not propose one-hot encoding.
- Do not propose missing-value handling.
- Do not propose arbitrary Python code.
- Feature engineering is handled by a separate Feature Engineering Agent.
- If another baseline model remains, choose continue rather than feature_engineering.
- Do not re-propose a baseline model that has already been executed.
- Do not repeatedly request the same feature engineering experiment.
- Use the actual validation metrics when deciding whether further experimentation is worthwhile.
- Do not request feature engineering merely because feature engineering is available.
- Prefer finish when the latest experiments show no meaningful improvement.

Return your decision based only on the actual experiment results provided.
"""


def critic_agent(
    state: ResearchState,
) -> dict:
    logger.info("--- CRITIC AGENT ---")

    research_plan = state.get(
        "research_plan",
        {},
    )

    experiments = state.get(
        "experiments",
        [],
    )

    feature_engineering_history = state.get(
        "feature_engineering_history",
        [],
    )

    critic_feedback_so_far = state.get(
        "critic_feedback",
        [],
    )

    experiment_history = state.get(
        "experiment_history",
        [],
    )

    feature_attempts = len(
        feature_engineering_history
    )

    problem_type = research_plan.get(
        "problem_type"
    )

    supported_models = get_supported_models(
        problem_type
    )

    logger.info(
        f"Feature engineering attempts: "
        f"{feature_attempts}/{MAX_FEATURE_EXPERIMENTS}"
    )

    prompt = CRITIC_PROMPT.format(
        research_plan=research_plan,
        experiments=summarize_experiments(experiments),
        feature_engineering_history=(
            feature_engineering_history
        ),
        critic_feedback=recent_feedback(
            critic_feedback_so_far
        ),
        allowed_parameters=ALLOWED_PARAMETERS,
    )

    logger.info(
        "Critic reviewing experiments..."
    )

    try:
        decision = structured_llm.invoke(
            prompt
        )

    except Exception as error:
        logger.error(
            f"Critic LLM call failed: {error}. "
            "Stopping the research loop."
        )

        critic_result = {
            "decision": "finish",
            "experiment_type": "none",
            "model": None,
            "parameters": {},
            "concerns": [
                f"Critic LLM call failed: {error}"
            ],
            "recommendation": (
                "Stopping due to an LLM call failure."
            ),
            "reasoning": (
                "The critic could not be consulted "
                "because the underlying LLM call failed "
                "or timed out."
            ),
        }

        critic_feedback = list(
            critic_feedback_so_far
        )

        critic_feedback.append(
            f"[error] Critic LLM call failed: {error}"
        )

        return {
            "critic_analysis": critic_result,
            "critic_feedback": critic_feedback,
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "parameters": {},
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "critic_agent": critic_result,
            },
            "current_task": (
                "Critic failed; stopping research loop"
            ),
        }

    if (
        decision.model
        and decision.model not in supported_models
    ):
        logger.warning(
            f"Critic proposed unsupported model "
            f"'{decision.model}'. "
            "Overriding decision to finish."
        )

        decision.decision = "finish"
        decision.experiment_type = "none"
        decision.model = None
        decision.parameters = {}

    if (
        decision.decision == "continue"
        and decision.experiment_type
        == "baseline_model"
    ):
        candidate_id = (
            f"baseline_model::{decision.model}"
            if decision.model
            else None
        )

        if (
            not decision.model
            or candidate_id in experiment_history
        ):
            logger.warning(
                "Critic proposed a duplicate or invalid "
                "baseline experiment. "
                "Overriding decision to finish."
            )

            decision.decision = "finish"
            decision.experiment_type = "none"
            decision.model = None
            decision.parameters = {}

    if (
        decision.decision
        == "feature_engineering"
    ):
        if (
            feature_attempts
            >= MAX_FEATURE_EXPERIMENTS
        ):
            logger.info(
                "Feature engineering budget exhausted. "
                "Forcing finish."
            )

            decision.decision = "finish"
            decision.experiment_type = "none"
            decision.model = None
            decision.parameters = {}

            decision.recommendation = (
                "Feature engineering budget exhausted."
            )

            decision.reasoning = (
                f"{MAX_FEATURE_EXPERIMENTS} feature "
                "engineering experiments have already "
                "been attempted."
            )

    logger.info(
        f"Decision: {decision.decision}"
    )

    logger.info(
        f"Experiment type: "
        f"{decision.experiment_type}"
    )

    logger.info(
        f"Model: {decision.model}"
    )

    logger.info(
        f"Recommendation: "
        f"{decision.recommendation}"
    )

    critic_result = {
        "decision": decision.decision,
        "experiment_type": (
            decision.experiment_type
        ),
        "model": decision.model,
        "parameters": decision.parameters,
        "concerns": decision.concerns,
        "recommendation": (
            decision.recommendation
        ),
        "reasoning": decision.reasoning,
    }

    feedback_entry = (
        f"[{decision.decision}/"
        f"{decision.experiment_type}"
        f"{'/' + decision.model if decision.model else ''}] "
        f"{decision.recommendation} -- "
        f"{decision.reasoning}"
    )

    critic_feedback = list(
        critic_feedback_so_far
    )

    critic_feedback.append(
        feedback_entry
    )

    return {
        "critic_analysis": critic_result,
        "critic_feedback": critic_feedback,
        "proposed_experiment": {
            "experiment_type": (
                decision.experiment_type
            ),
            "model": decision.model,
            "parameters": decision.parameters,
        },
        "agent_results": {
            **state.get(
                "agent_results",
                {},
            ),
            "critic_agent": critic_result,
        },
        "current_task": (
            "Critic decision completed"
        ),
    }

