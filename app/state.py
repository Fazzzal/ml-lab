from typing import TypedDict, Any


class ResearchState(TypedDict, total=False):
    dataset_path: str
    target_column: str

    dataset_info: dict[str, Any]
    feature_analysis: dict[str, Any]
    research_plan: dict[str, Any]
    evaluation: dict[str, Any]
    critic_analysis: dict[str, Any]

    proposed_experiment: dict[str, Any]
    feature_proposal: dict[str, Any]

    agent_results: dict[str, Any]

    current_task: str
    next_agent: str
    supervisor_reason: str

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

    experiments: list[dict[str, Any]]
    experiment_history: list[str]
    feature_engineering_history: list[str]
    critic_feedback: list[str]

    retry_count: int
    feature_attempts: int

    best_experiment: dict[str, Any]
    best_validation_metric: float
    best_metric_name: str

    final_report: str