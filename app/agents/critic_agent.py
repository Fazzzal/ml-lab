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

All baseline models from the research plan have already been evaluated.

Your job is now to examine the completed experiments and decide whether
feature engineering is worthwhile or whether the research should finish.

Research plan:

{research_plan}

Completed experiments:

{experiments}

Current best experiment:

{best_experiment}

Feature engineering attempts already made:

{feature_engineering_history}

Recent feedback from previous decisions:

{critic_feedback}

Allowed hyperparameters per model:

{allowed_parameters}

You have two relevant decisions.

1. feature_engineering

Use this only when:
- all baseline models have already been evaluated
- fewer than the allowed number of feature engineering experiments have
  been attempted
- the completed experiments provide a reasonable opportunity to test a
  useful feature transformation
- the latest experiments do not already show that further experimentation
  is unlikely to help

2. finish

Use this when:
- the feature engineering budget has been exhausted
- recent feature engineering experiments have not provided meaningful
  improvement
- there is no useful remaining experiment
- or the available evidence does not justify another experiment

Important rules:

- Do not propose baseline models.
- Do not propose unsupported models.
- Only propose hyperparameters listed in the allowed set.
- Do not propose scaling.
- Do not propose one-hot encoding.
- Do not propose missing-value handling.
- Do not propose arbitrary Python code.
- Feature engineering is handled by a separate Feature Engineering Agent.
- Do not repeatedly request the same feature engineering experiment.
- Use the actual validation metrics when deciding whether further
  experimentation is worthwhile.
- Do not request feature engineering merely because feature engineering
  is available.
- Prefer finish when recent experiments show no meaningful improvement.

Return your decision based only on the actual experiment results provided.
"""


def _get_completed_baseline_models(
    experiments: list[dict],
) -> set[str]:
    completed_models = set()

    for experiment in experiments:
        if experiment.get("experiment_type") != "baseline_model":
            continue

        model = experiment.get("model")

        if model:
            completed_models.add(model)

    return completed_models


def _get_pending_baseline_model(
    research_plan: dict,
    experiments: list[dict],
) -> str | None:
    planned_models = research_plan.get(
        "models",
        [],
    )

    completed_models = _get_completed_baseline_models(
        experiments
    )

    for model in planned_models:
        if model not in completed_models:
            return model

    return None


def _build_baseline_decision(
    model: str,
    completed_models: set[str],
) -> dict:
    return {
        "decision": "continue",
        "experiment_type": "baseline_model",
        "model": model,
        "parameters": {},
        "concerns": [
            (
                f"Baseline models completed so far: "
                f"{sorted(completed_models)}"
            )
        ],
        "recommendation": (
            f"Run the {model} baseline model before "
            "considering feature engineering."
        ),
        "reasoning": (
            f"The research plan requires evaluation of all "
            f"planned baseline models. The {model} baseline "
            "has not yet been executed, so it must be evaluated "
            "before feature engineering or termination."
        ),
    }


def _build_finish_decision(
    reason: str,
) -> dict:
    return {
        "decision": "finish",
        "experiment_type": "none",
        "model": None,
        "parameters": {},
        "concerns": [
            reason
        ],
        "recommendation": (
            "No further baseline experiment is required."
        ),
        "reasoning": reason,
    }


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

    problem_type = research_plan.get(
        "problem_type"
    )

    if not problem_type:
        logger.error(
            "Research plan does not contain a problem type."
        )

        critic_result = _build_finish_decision(
            "Cannot determine problem type from research plan."
        )

        return {
            "critic_analysis": critic_result,
            "proposed_experiment": {
                "experiment_type": "none",
                "model": None,
                "parameters": {},
            },
            "current_task": (
                "Critic stopped because problem type is missing"
            ),
        }

    supported_models = get_supported_models(
        problem_type
    )

    feature_attempts = len(
        feature_engineering_history
    )

    completed_models = _get_completed_baseline_models(
        experiments
    )

    pending_model = _get_pending_baseline_model(
        research_plan,
        experiments
    )

    logger.info(
        f"Feature engineering attempts: "
        f"{feature_attempts}/{MAX_FEATURE_EXPERIMENTS}"
    )

    logger.info(
        f"Completed baseline models: "
        f"{sorted(completed_models)}"
    )

    logger.info(
        f"Pending baseline model: "
        f"{pending_model}"
    )

    if pending_model is not None:
        if pending_model not in supported_models:
            logger.warning(
                f"Research plan contains unsupported model: "
                f"{pending_model}"
            )

            critic_result = _build_finish_decision(
                (
                    f"The research plan requested unsupported "
                    f"model: {pending_model}"
                )
            )

        else:
            logger.info(
                f"Deterministically selecting pending baseline: "
                f"{pending_model}"
            )

            critic_result = _build_baseline_decision(
                model=pending_model,
                completed_models=completed_models,
            )

        critic_feedback = list(
            critic_feedback_so_far
        )

        feedback_entry = (
            f"[{critic_result['decision']}/"
            f"{critic_result['experiment_type']}/"
            f"{critic_result['model']}] "
            f"{critic_result['recommendation']} -- "
            f"{critic_result['reasoning']}"
        )

        critic_feedback.append(
            feedback_entry
        )

        logger.info(
            f"Decision: {critic_result['decision']}"
        )

        logger.info(
            f"Experiment type: "
            f"{critic_result['experiment_type']}"
        )

        logger.info(
            f"Model: {critic_result['model']}"
        )

        logger.info(
            f"Recommendation: "
            f"{critic_result['recommendation']}"
        )

        return {
            "critic_analysis": critic_result,
            "critic_feedback": critic_feedback,
            "proposed_experiment": {
                "experiment_type": (
                    critic_result["experiment_type"]
                ),
                "model": critic_result["model"],
                "parameters": critic_result["parameters"],
            },
            "agent_results": {
                **state.get(
                    "agent_results",
                    {},
                ),
                "critic_agent": critic_result,
            },
            "current_task": (
                "Pending baseline selected"
            ),
        }

    if feature_attempts >= MAX_FEATURE_EXPERIMENTS:
        logger.info(
            "All baseline models completed and feature "
            "engineering budget exhausted. Forcing finish."
        )

        critic_result = _build_finish_decision(
            (
                f"All baseline models have been evaluated and "
                f"the feature engineering budget of "
                f"{MAX_FEATURE_EXPERIMENTS} experiments "
                "has been exhausted."
            )
        )

        critic_feedback = list(
            critic_feedback_so_far
        )

        critic_feedback.append(
            f"[finish/none] "
            f"{critic_result['recommendation']} -- "
            f"{critic_result['reasoning']}"
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
                "Baseline research complete; "
                "feature budget exhausted"
            ),
        }

    best_experiment = state.get(
        "best_experiment",
        {},
    )

    prompt = CRITIC_PROMPT.format(
        research_plan=research_plan,
        experiments=summarize_experiments(
            experiments
        ),
        best_experiment=best_experiment,
        feature_engineering_history=(
            feature_engineering_history
        ),
        critic_feedback=recent_feedback(
            critic_feedback_so_far
        ),
        allowed_parameters=ALLOWED_PARAMETERS,
    )

    logger.info(
        "All baseline models completed. "
        "Critic reviewing feature engineering opportunity..."
    )

    try:
        decision = structured_llm.invoke(
            prompt
        )

    except Exception as error:
        logger.error(
            f"Critic LLM call failed: {error}"
        )

        critic_result = _build_finish_decision(
            (
                "All baseline models were completed, but "
                "the Critic LLM failed while evaluating "
                "whether further feature engineering "
                "was worthwhile."
            )
        )

        critic_result["concerns"] = [
            f"Critic LLM call failed: {error}"
        ]

        critic_result["recommendation"] = (
            "Stopping safely because the feature-engineering "
            "decision could not be evaluated."
        )

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
                "Critic failed during feature-engineering "
                "decision; research stopped safely"
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

        decision.recommendation = (
            "Stopping because the proposed model "
            "is not supported."
        )

        decision.reasoning = (
            "The Critic proposed a model that is not "
            "supported by the execution layer."
        )

    if decision.decision == "continue":
        logger.warning(
            "Critic attempted to request another baseline "
            "after all baselines were already completed. "
            "Overriding decision to finish."
        )

        decision.decision = "finish"
        decision.experiment_type = "none"
        decision.model = None
        decision.parameters = {}

        decision.recommendation = (
            "All baseline models have already been evaluated."
        )

        decision.reasoning = (
            "No unexecuted baseline model remains in "
            "the research plan."
        )

    if decision.decision == "feature_engineering":
        if feature_attempts >= MAX_FEATURE_EXPERIMENTS:
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

    if decision.decision == "finish":
        decision.experiment_type = "none"
        decision.model = None
        decision.parameters = {}

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