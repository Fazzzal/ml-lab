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

    experiments: list[dict[str, Any]]
    experiment_history: list[str]
    feature_engineering_history: list[str]
    critic_feedback: list[str]

    # Fixed 3-way split, computed once (split_agent) right after
    # the problem type is known. experiment_agent always trains
    # on train_indices and reports metrics on val_indices, so the
    # critic loop never sees test_indices until final_report_agent
    # touches it exactly once at the end.
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

    held_out_evaluation: dict[str, Any]

    retry_count: int
    final_report: str
