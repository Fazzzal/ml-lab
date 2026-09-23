
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.state import ResearchState
from app.config import settings
from app.tools.ml_tools import (
    get_supported_models,
    ALLOWED_PARAMETERS,
)
from app.tools.feature_tools import TRANSFORMATIONS
from app.tools.summarize import (
    summarize_experiments,
    recent_feedback,
)
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

    model: str | None = None

    feature_type: str | None = None

    parameters: dict = {}

    concerns: list[str] = []

    recommendation: str =""

    reasoning: str = ""


structured_llm = llm.with_structured_output(
    CriticDecision
)


CRITIC_PROMPT = """
You are the Critic Agent in an autonomous machine learning research system.

All baseline models from the research plan have already been evaluated.

Your job is now to decide whether to run another feature engineering
experiment or finish the research.

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

Supported models for this problem:

{supported_models}

Supported feature transformations:

{supported_feature_types}

You have two valid decisions.

1. feature_engineering

Use this when:
- all baseline models have already been evaluated
- fewer than {max_feature_experiments} feature engineering experiments
  have been executed
- there is a useful unexplored model/transformation combination

For feature_engineering you MUST return:
- experiment_type = "feature_engineering"
- a supported model
- an exact feature_type from the supported feature transformations

2. finish

Use this when:
- the feature engineering budget is exhausted
- there is no useful remaining experiment
- or the evidence strongly suggests further experimentation is not useful

Important rules:

- Do NOT return experiment_type = "baseline_model".
- Do NOT return decision = "continue" after all baseline models are complete.
- Do NOT propose unsupported models.
- Do NOT invent feature transformation names.
- Do NOT propose scaling.
- Do NOT propose one-hot encoding.
- Do NOT propose missing-value handling.
- Do NOT propose arbitrary Python code.
- Do NOT repeatedly request an already-tested model/transformation combination.
- Use actual validation metrics when deciding.
- Prefer a different model when previous feature engineering attempts all used
  the same model.
- If feature engineering is selected, always provide both model and feature_type.
- Feature source columns are selected by the Feature Engineering Agent.

Return only the structured decision.
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


def _get_feature_attempts(
    experiments: list[dict],
) -> int:
    return sum(
        1
        for experiment in experiments
        if experiment.get("experiment_type")
        == "feature_engineering"
    )


def _get_tested_feature_combinations(
    experiments: list[dict],
) -> set[tuple[str, str]]:
    tested = set()

    for experiment in experiments:
        if experiment.get("experiment_type") != "feature_engineering":
            continue

        model = experiment.get("model")
        feature_type = experiment.get("feature_type")

        if model and feature_type:
            tested.add(
                (
                    model,
                    feature_type,
                )
            )

    return tested


def _select_fallback_feature_experiment(
    supported_models: list[str],
    experiments: list[dict],
    best_model: str | None,
) -> tuple[str | None, str | None]:
    tested = _get_tested_feature_combinations(
        experiments
    )

    if not supported_models:
        return None, None

    model_order = list(supported_models)

    if best_model in model_order:
        model_order.remove(best_model)
        model_order.append(best_model)

    for model in model_order:
        for feature_type in TRANSFORMATIONS:
            if (
                model,
                feature_type,
            ) not in tested:
                return model, feature_type

    return None, None


def _build_baseline_decision(
    model: str,
    completed_models: set[str],
) -> dict:
    return {
        "decision": "continue",
        "experiment_type": "baseline_model",
        "model": model,
        "feature_type": None,
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
            "has not yet been executed."
        ),
    }


def _build_feature_decision(
    model: str,
    feature_type: str,
    reason: str,
) -> dict:
    return {
        "decision": "feature_engineering",
        "experiment_type": "feature_engineering",
        "model": model,
        "feature_type": feature_type,
        "parameters": {},
        "concerns": [],
        "recommendation": (
            f"Run {feature_type} feature engineering "
            f"with {model}."
        ),
        "reasoning": reason,
    }


def _build_finish_decision(
    reason: str,
) -> dict:
    return {
        "decision": "finish",
        "experiment_type": "none",
        "model": None,
        "feature_type": None,
        "parameters": {},
        "concerns": [
            reason
        ],
        "recommendation": (
            "No further experiment is required."
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
                "feature_type": None,
                "parameters": {},
            },
            "current_task": (
                "Critic stopped because problem type is missing"
            ),
        }

    supported_models = get_supported_models(
        problem_type
    )

    feature_attempts = _get_feature_attempts(
        experiments
    )

    completed_models = _get_completed_baseline_models(
        experiments
    )

    pending_model = _get_pending_baseline_model(
        research_plan,
        experiments,
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

        return {
            "critic_analysis": critic_result,
            "critic_feedback": critic_feedback,
            "proposed_experiment": {
                "experiment_type": (
                    critic_result["experiment_type"]
                ),
                "model": critic_result["model"],
                "feature_type": (
                    critic_result["feature_type"]
                ),
                "parameters": (
                    critic_result["parameters"]
                ),
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
            "All baselines completed and feature "
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
                "feature_type": None,
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
        supported_models=supported_models,
        supported_feature_types=list(
            TRANSFORMATIONS.keys()
        ),
        max_feature_experiments=(
            MAX_FEATURE_EXPERIMENTS
        ),
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

        fallback_model, fallback_feature_type = (

            _select_fallback_feature_experiment(

                supported_models=supported_models,

                experiments=experiments,

                best_model=best_experiment.get("model"),

            )

        )

        if (

                fallback_model is not None

                and fallback_feature_type is not None

                and feature_attempts < MAX_FEATURE_EXPERIMENTS

        ):
            logger.info(

                "Critic LLM failed. Falling back to "

                "deterministic feature engineering."

            )

            critic_result = _build_feature_decision(

                model=fallback_model,

                feature_type=fallback_feature_type,

                reason=(

                    "The Critic LLM failed, so the system selected "

                    "an untested model/transformation combination "

                    "deterministically."

                ),

            )

            critic_result["concerns"] = [

                f"Critic LLM call failed: {error}"

            ]

            critic_result["recommendation"] = (

                f"Run {fallback_feature_type} "

                f"feature engineering with {fallback_model}."

            )

            critic_feedback = list(

                critic_feedback_so_far

            )

            critic_feedback.append(

                f"[error/feature_engineering/"

                f"{fallback_model}/{fallback_feature_type}] "

                f"Critic LLM failed; deterministic fallback "

                f"selected. Error: {error}"

            )

            return {

                "critic_analysis": critic_result,

                "critic_feedback": critic_feedback,

                "proposed_experiment": {

                    "experiment_type": "feature_engineering",

                    "model": fallback_model,

                    "feature_type": fallback_feature_type,

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

                    "Critic LLM failed; deterministic "

                    "feature-engineering fallback selected"

                ),

            }

        critic_result = _build_finish_decision(

            (

                "The Critic LLM failed and no valid "

                "feature-engineering fallback remains."

            )

        )

        critic_result["concerns"] = [

            f"Critic LLM call failed: {error}"

        ]

        critic_feedback = list(

            critic_feedback_so_far

        )

        critic_feedback.append(

            f"[error/finish] Critic LLM failed: {error}"

        )

        return {

            "critic_analysis": critic_result,

            "critic_feedback": critic_feedback,

            "proposed_experiment": {

                "experiment_type": "none",

                "model": None,

                "feature_type": None,

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

                "Critic failed and no deterministic "

                "fallback remained"

            ),

        }

    if (
        decision.model
        and decision.model not in supported_models
    ):
        logger.warning(
            f"Critic proposed unsupported model "
            f"'{decision.model}'. "
            "Replacing with a deterministic valid choice."
        )

        decision.model = None

    if (
        decision.feature_type
        and decision.feature_type
        not in TRANSFORMATIONS
    ):
        logger.warning(
            f"Critic proposed unsupported feature type "
            f"'{decision.feature_type}'. "
            "Replacing with a deterministic valid choice."
        )

        decision.feature_type = None

    if decision.decision == "continue":
        logger.warning(
            "Critic requested another baseline after "
            "all baselines were completed."
        )

        fallback_model, fallback_feature_type = (
            _select_fallback_feature_experiment(
                supported_models=supported_models,
                experiments=experiments,
                best_model=best_experiment.get(
                    "model"
                ),
            )
        )

        if (
            fallback_model is not None
            and fallback_feature_type is not None
        ):
            logger.info(
                "Converting invalid baseline continuation "
                "into feature engineering."
            )

            decision = CriticDecision(
                decision="feature_engineering",
                experiment_type="feature_engineering",
                model=fallback_model,
                feature_type=fallback_feature_type,
                parameters={},
                concerns=[
                    (
                        "The Critic requested another baseline "
                        "after all baselines were completed. "
                        "A valid unexplored feature/model "
                        "combination was selected instead."
                    )
                ],
                recommendation=(
                    f"Run {fallback_feature_type} "
                    f"feature engineering with "
                    f"{fallback_model}."
                ),
                reasoning=(
                    "All baseline models are complete and "
                    "feature engineering budget remains. "
                    "The system selected an untested "
                    "model/transformation combination."
                ),
            )

    if decision.decision == "feature_engineering":
        if not decision.model:
            fallback_model, fallback_feature_type = (
                _select_fallback_feature_experiment(
                    supported_models=supported_models,
                    experiments=experiments,
                    best_model=best_experiment.get(
                        "model"
                    ),
                )

                )

            decision.model = fallback_model

            if not decision.feature_type:
                decision.feature_type = (
                    fallback_feature_type
                )

        if (
            not decision.model
            or decision.model not in supported_models
            or not decision.feature_type
            or decision.feature_type
            not in TRANSFORMATIONS
        ):
            logger.warning(
                "Critic returned an invalid feature-engineering "
                "decision. Selecting a deterministic valid "
                "combination."
            )

            fallback_model, fallback_feature_type = (
                _select_fallback_feature_experiment(
                    supported_models=supported_models,
                    experiments=experiments,
                    best_model=best_experiment.get(
                        "model"
                    ),
                )
            )

            if (
                fallback_model is None
                or fallback_feature_type is None
            ):
                decision.decision = "finish"
                decision.experiment_type = "none"
                decision.model = None
                decision.feature_type = None
                decision.parameters = {}

                decision.recommendation = (
                    "No untested feature-engineering "
                    "combination remains."
                )

                decision.reasoning = (
                    "Every supported model/transformation "
                    "combination available to the execution "
                    "layer has already been tested."
                )

            else:
                decision.decision = (
                    "feature_engineering"
                )
                decision.experiment_type = (
                    "feature_engineering"
                )
                decision.model = fallback_model
                decision.feature_type = (
                    fallback_feature_type
                )
                decision.parameters = {}

    if decision.decision == "finish":
        decision.experiment_type = "none"
        decision.model = None
        decision.feature_type = None
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
        f"Feature type: "
        f"{decision.feature_type}"
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
        "feature_type": decision.feature_type,
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
        f"{'/' + decision.model if decision.model else ''}"
        f"{'/' + decision.feature_type if decision.feature_type else ''}] "
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
            "feature_type": (
                decision.feature_type
            ),
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

