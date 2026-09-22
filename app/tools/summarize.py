def summarize_experiments(experiments: list[dict]) -> list[dict]:
    """Condense full experiment result dicts (which carry nested
    cross-validation stats, applied/rejected params, sample
    counts) down to just what the critic/feature-engineering LLM
    prompts need in order to reason about what's already been
    tried. Without this, prompt size — and therefore latency —
    grows with every experiment run in a session."""

    summary = []

    for experiment in experiments:
        metric_name = "f1" if "f1" in experiment else "rmse"

        summary.append({
            "model": experiment.get("model"),
            "type": experiment.get("experiment_type"),
            "feature_type": experiment.get("feature_type"),
            metric_name: experiment.get(metric_name),
        })

    return summary


def recent_feedback(critic_feedback: list[str], limit: int = 5) -> list[str]:
    """Only the most recent N feedback entries — the full history
    is still preserved in state (and in final_report_agent's
    output), this just bounds what goes into every LLM prompt."""

    if len(critic_feedback) <= limit:
        return critic_feedback

    omitted = len(critic_feedback) - limit

    return (
        [f"...({omitted} earlier entries omitted)..."]
        + critic_feedback[-limit:]
    )