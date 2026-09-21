from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.state import ResearchState
from app.config import settings
from app.tools.ml_tools import get_supported_models, ALLOWED_PARAMETERS
from app.logging_config import get_logger

logger = get_logger(__name__)


llm = ChatGroq(
    model=settings.GROQ_MODEL
)


class CriticDecision(BaseModel):
    decision: Literal[
        "continue",
        "feature_engineering",
        "finish"
    ]

    experiment_type: Literal[
        "baseline_model",
        "feature_engineering",
        "none"
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
You are the Critic Agent in an autonomous
machine learning research system.

Your job is to examine the experiments completed
so far and decide what should happen next.

The baseline models in the research plan are:

{research_plan}

Completed experiments (metrics are on the validation split):

{experiments}

Feature engineering attempts already made:

{feature_engineering_history}

Allowed hyperparameters per model, if you want to propose a tuned
re-run of a model that hasn't been tried with these values yet.
Any parameter not listed here will be silently ignored by the
execution layer, so don't bother proposing it:

{allowed_parameters}

Your decisions are:

1. continue

Use this when another baseline model from the
research plan still needs to be evaluated.

2. feature_engineering

Use this when all baseline models have been
evaluated and there is a reasonable opportunity
to improve the model through feature engineering.

3. finish

Use this when the available experiments have been
completed and there is no useful remaining
experiment.

Important rules:

- Do not propose unsupported models.
- Only propose hyperparameters listed in the allowed set above.
- Do not propose scaling.
- Do not propose one-hot encoding.
- Do not propose missing-value handling.
- Do not propose arbitrary Python code.
- Feature engineering will be handled by a separate
  Feature Engineering Agent.
- If another baseline model remains, choose
  continue rather than feature_engineering.
- Once all baselines are complete, feature engineering
  may be requested.
- Do not repeatedly request feature engineering if
  previous attempts have already been exhausted.
- Do not re-propose a baseline model + parameter combination
  that has already appeared in the completed experiments above.

Return your reasoning based on the actual results.
"""


def critic_agent(
    state: ResearchState
) -> dict:
    logger.info("--- CRITIC AGENT ---")

    research_plan = state.get(
        "research_plan",
        {}
    )

    experiments = state.get(
        "experiments",
        []
    )

    feature_engineering_history = state.get(
        "feature_engineering_history",
        []
    )

    prompt = CRITIC_PROMPT.format(
        research_plan=research_plan,
        experiments=experiments,
        feature_engineering_history=(
            feature_engineering_history
        ),
        allowed_parameters=ALLOWED_PARAMETERS,
    )

    logger.info("Critic reviewing experiments...")

    decision = structured_llm.invoke(
        prompt
    )

    experiment_history = state.get(
        "experiment_history",
        []
    )

    problem_type = research_plan.get(
        "problem_type"
    )

    supported_models = get_supported_models(
        problem_type
    )

    # Guard against the LLM hallucinating a model name that
    # doesn't exist in the execution layer (e.g. "XGBoost", a
    # misspelling, or a model from the wrong problem type). The
    # `model` field is free text, not a Literal, so nothing else
    # enforces this before it reaches sklearn.
    if (
        decision.model
        and decision.model not in supported_models
    ):
        logger.warning(
            f"Critic proposed an unsupported model "
            f"'{decision.model}'; overriding decision to finish."
        )

        decision.decision = "finish"
        decision.experiment_type = "none"
        decision.model = None
        decision.parameters = {}

    # Guard against the LLM re-proposing a baseline that has
    # already been run. Without this, a non-deterministic model
    # can keep asking for the same experiment every cycle and the
    # graph never terminates on its own.
    if (
        decision.decision == "continue"
        and decision.experiment_type == "baseline_model"
    ):
        candidate_id = (
            f"baseline_model::{decision.model}"
            if decision.model else None
        )

        if not decision.model or candidate_id in experiment_history:
            logger.warning(
                "Critic proposed a duplicate/invalid baseline "
                "experiment; overriding decision to finish."
            )

            decision.decision = "finish"
            decision.experiment_type = "none"
            decision.model = None
            decision.parameters = {}

    logger.info(f"Decision: {decision.decision}")
    logger.info(f"Experiment type: {decision.experiment_type}")
    logger.info(f"Model: {decision.model}")
    logger.info(f"Recommendation: {decision.recommendation}")

    critic_result = {
        "decision": decision.decision,
        "experiment_type": (
            decision.experiment_type
        ),
        "model": decision.model,
        "parameters": decision.parameters,
        "concerns": decision.concerns,
        "recommendation": decision.recommendation,
        "reasoning": decision.reasoning,
    }

    feedback_entry = (
        f"[{decision.decision}/{decision.experiment_type}"
        f"{'/' + decision.model if decision.model else ''}] "
        f"{decision.recommendation} -- {decision.reasoning}"
    )

    critic_feedback = list(
        state.get("critic_feedback", [])
    )
    critic_feedback.append(feedback_entry)

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
                {}
            ),
            "critic_agent": critic_result,
        },
        "current_task": (
            "Critic decision completed"
        ),
    }
