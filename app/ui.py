import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.graph import build_graph


st.set_page_config(
    page_title="ML Research Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    defaults = {
        "run_state": None,
        "running": False,
        "dataset_path": None,
        "target_column": None,
        "run_completed": False,
        "run_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_value(state: dict[str, Any], key: str, default=None):
    value = state.get(key, default)

    if value is None:
        return default

    return value


def format_metric(value):
    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def get_experiments(state):
    experiments = state.get("experiments", [])

    if not isinstance(experiments, list):
        return []

    return experiments


def get_agent_status(state):
    experiments = get_experiments(state)

    dataset_info = state.get("dataset_info")
    feature_analysis = state.get("feature_analysis")
    research_plan = state.get("research_plan")
    evaluation = state.get("evaluation")
    critic_analysis = state.get("critic_analysis")
    final_report = state.get("final_report")

    statuses = {
        "Data Agent": bool(dataset_info),
        "Feature Agent": bool(feature_analysis),
        "Research Agent": bool(research_plan),
        "Split Agent": bool(
            state.get("train_indices")
            and state.get("val_indices")
            and state.get("test_indices")
        ),
        "Experiment Agent": len(experiments) > 0,
        "Evaluation Agent": bool(evaluation),
        "Critic Agent": bool(critic_analysis),
        "Final Report": bool(final_report),
    }

    return statuses


def render_agent_status(state):
    st.subheader("Research Pipeline")

    statuses = get_agent_status(state)

    agents = [
        "Data Agent",
        "Feature Agent",
        "Research Agent",
        "Split Agent",
        "Experiment Agent",
        "Evaluation Agent",
        "Critic Agent",
        "Final Report",
    ]

    cols = st.columns(len(agents))

    for col, agent in zip(cols, agents):
        with col:
            if statuses[agent]:
                st.success(f"✓ {agent}")
            else:
                st.empty()
                st.caption(f"○ {agent}")


def render_dataset_overview(state):
    dataset_info = state.get("dataset_info", {})

    if not dataset_info:
        return

    st.subheader("Dataset Overview")

    rows = dataset_info.get("rows", 0)
    columns = dataset_info.get("columns", 0)

    numeric_columns = dataset_info.get("numeric_columns", [])
    categorical_columns = dataset_info.get("categorical_columns", [])
    missing_values = dataset_info.get("missing_values", {})

    research_plan = state.get("research_plan", {})
    problem_type = research_plan.get("problem_type", "Unknown")

    missing_count = sum(missing_values.values())

    cols = st.columns(6)

    metrics = [
        ("Rows", rows),
        ("Columns", columns),
        ("Problem", problem_type.title()),
        ("Numeric", len(numeric_columns)),
        ("Categorical", len(categorical_columns)),
        ("Missing Values", missing_count),
    ]

    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_feature_analysis(state):
    feature_analysis = state.get("feature_analysis", {})

    if not feature_analysis:
        return

    with st.expander("Feature Analysis", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Identifier columns**")
            st.write(feature_analysis.get("identifier_columns", []))

            st.write("**Numeric features**")
            st.write(feature_analysis.get("numeric_features", []))

        with col2:
            st.write("**Categorical features**")
            st.write(feature_analysis.get("categorical_features", []))

            st.write("**Missing-value columns**")
            st.write(feature_analysis.get("missing_value_columns", []))

        recommendations = feature_analysis.get("recommendations", [])

        if recommendations:
            st.write("**Recommendations**")
            st.dataframe(
                pd.DataFrame(recommendations),
                use_container_width=True,
                hide_index=True,
            )


def render_research_plan(state):
    research_plan = state.get("research_plan", {})

    if not research_plan:
        return

    with st.expander("Research Plan", expanded=False):
        st.write(
            f"**Problem type:** "
            f"{research_plan.get('problem_type', 'Unknown').title()}"
        )

        st.write("**Models:**")
        for model in research_plan.get("models", []):
            st.write(f"- {model}")

        reasoning = research_plan.get("reasoning")

        if reasoning:
            st.write("**Reasoning**")
            st.write(reasoning)


def experiment_dataframe(state):
    experiments = get_experiments(state)

    if not experiments:
        return pd.DataFrame()

    rows = []

    for experiment in experiments:
        rows.append(
            {
                "Model": experiment.get("model", "Unknown"),
                "Type": experiment.get(
                    "experiment_type",
                    "Unknown",
                ),
                "Metric": experiment.get(
                    "metric_name",
                    "Unknown",
                ),
                "Validation": experiment.get(
                    "validation_metric"
                ),
                "Accuracy": experiment.get("accuracy"),
                "Feature": experiment.get(
                    "feature_name",
                    "",
                ),
                "Status": (
                    "★ Best"
                    if experiment.get("is_best")
                    else "Completed"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_experiments(state):
    st.subheader("Experiments")

    df = experiment_dataframe(state)

    if df.empty:
        st.info("No experiments have been completed yet.")
        return

    display_df = df.copy()

    for column in ["Validation", "Accuracy"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: (
                    f"{value:.4f}"
                    if isinstance(value, (float, int))
                    else value
                )
            )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def render_current_best(state):
    evaluation = state.get("evaluation", {})

    if not evaluation:
        return

    st.subheader("Current Best Model")

    best_model = evaluation.get("best_model")
    best_metric = evaluation.get("best_metric")
    metric_name = evaluation.get("metric_name")

    cols = st.columns(3)

    cols[0].metric(
        "Best Model",
        best_model or "—",
    )

    cols[1].metric(
        "Validation Metric",
        format_metric(best_metric),
    )

    cols[2].metric(
        "Metric",
        metric_name.upper() if metric_name else "—",
    )


def render_critic(state):
    critic = state.get("critic_analysis", {})

    if not critic:
        return

    st.subheader("Critic")

    decision = critic.get("decision", "unknown")
    experiment_type = critic.get("experiment_type", "none")
    model = critic.get("model")
    feature_type = critic.get("feature_type")
    recommendation = critic.get("recommendation")
    reasoning = critic.get("reasoning")
    concerns = critic.get("concerns", [])

    cols = st.columns(4)

    cols[0].metric(
        "Decision",
        decision.replace("_", " ").title(),
    )

    cols[1].metric(
        "Experiment Type",
        experiment_type.replace("_", " ").title(),
    )

    cols[2].metric(
        "Model",
        model or "—",
    )

    cols[3].metric(
        "Feature Type",
        feature_type or "—",
    )

    if recommendation:
        st.info(recommendation)

    if reasoning:
        with st.expander("Critic reasoning", expanded=False):
            st.write(reasoning)

    if concerns:
        with st.expander("Concerns", expanded=False):
            for concern in concerns:
                st.write(f"- {concern}")


def render_final_evaluation(state):
    evaluation = state.get("evaluation", {})

    if not evaluation:
        return

    held_out = evaluation.get("held_out_test")

    if not held_out:
        return

    st.subheader("Held-Out Test Evaluation")

    cols = st.columns(4)

    cols[0].metric(
        "Final Model",
        held_out.get("model", "—"),
    )

    cols[1].metric(
        "Test Accuracy",
        format_metric(held_out.get("accuracy")),
    )

    cols[2].metric(
        "Test F1",
        format_metric(held_out.get("f1")),
    )

    cols[3].metric(
        "Test Samples",
        held_out.get("test_samples", "—"),
    )

    st.caption(
        "The final model was retrained on train + validation "
        "and evaluated once on the held-out test set."
    )


def render_final_report(state):
    report = state.get("final_report")

    if not report:
        return

    st.subheader("Final Research Report")

    st.markdown(report)

    st.download_button(
        label="Download Report",
        data=report,
        file_name="ml_research_report.md",
        mime="text/markdown",
    )


def render_experiment_history(state):
    history = state.get("experiment_history", [])

    if not history:
        return

    with st.expander("Experiment History", expanded=False):
        for item in history:
            st.write(item)


def render_critic_feedback(state):
    feedback = state.get("critic_feedback", [])

    if not feedback:
        return

    with st.expander("Critic Feedback Trail", expanded=False):
        for item in feedback:
            st.write(item)


def save_uploaded_dataset(uploaded_file):
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    destination = data_dir / uploaded_file.name

    with open(destination, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return str(destination)


def load_uploaded_preview(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return None


def run_research(dataset_path, target_column):
    graph = build_graph()

    initial_state = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "experiments": [],
        "experiment_history": [],
        "feature_engineering_history": [],
        "critic_feedback": [],
        "retry_count": 0,
        "feature_attempts": 0,
        "agent_results": {},
    }

    config = {
        "configurable": {
            "thread_id": "streamlit_ml_research_lab"
        }
    }

    return graph.invoke(
        initial_state,
        config=config,
    )


def sidebar():
    st.sidebar.title("🧪 ML Research Lab")

    st.sidebar.caption(
        "Autonomous multi-agent machine learning experimentation"
    )

    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
    )

    dataset_path = None
    target_column = None

    if uploaded_file is not None:
        preview = load_uploaded_preview(uploaded_file)

        if preview is not None:
            st.sidebar.success(
                f"Loaded {len(preview):,} rows × "
                f"{len(preview.columns)} columns"
            )

            target_column = st.sidebar.selectbox(
                "Target column",
                options=list(preview.columns),
            )

            st.sidebar.caption(
                f"Target: `{target_column}`"
            )

            if st.sidebar.button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True,
            ):
                dataset_path = save_uploaded_dataset(
                    uploaded_file
                )

                st.session_state.dataset_path = dataset_path
                st.session_state.target_column = target_column
                st.session_state.running = True
                st.session_state.run_completed = False
                st.session_state.run_error = None
                st.session_state.run_state = None

                st.rerun()

    st.sidebar.divider()

    st.sidebar.write("**Current dataset**")

    if st.session_state.dataset_path:
        st.sidebar.caption(
            Path(
                st.session_state.dataset_path
            ).name
        )

    if st.session_state.target_column:
        st.sidebar.caption(
            f"Target: {st.session_state.target_column}"
        )

    if st.session_state.running:
        st.sidebar.warning("Research running...")

    elif st.session_state.run_completed:
        st.sidebar.success("Research completed")


def main():
    initialize_session_state()

    sidebar()

    st.title("🧪 ML Research Lab")

    st.caption(
        "Autonomous multi-agent ML experimentation powered by LangGraph"
    )

    if st.session_state.run_error:
        st.error(
            f"Research failed: {st.session_state.run_error}"
        )

    if (
        st.session_state.running
        and st.session_state.dataset_path
        and st.session_state.target_column
    ):
        st.info(
            "Running the autonomous research pipeline..."
        )

        progress = st.progress(0)

        try:
            state = run_research(
                st.session_state.dataset_path,
                st.session_state.target_column,
            )

            progress.progress(100)

            st.session_state.run_state = state
            st.session_state.running = False
            st.session_state.run_completed = True

            st.rerun()

        except Exception as exc:
            st.session_state.running = False
            st.session_state.run_error = str(exc)

            st.exception(exc)

            return

    state = st.session_state.run_state

    if state is None:
        st.info(
            "Upload a CSV dataset from the sidebar, "
            "select the target column, and start research."
        )

        st.markdown(
            """
            ### Pipeline

            **Data → Features → Research Plan → Split → "
            Experiment → Evaluation → Critic → Feature Engineering → Final Report**

            The Streamlit interface is only the presentation layer.
            The actual ML decisions and experiment execution remain inside
            the LangGraph pipeline.
            """
        )

        return

    render_agent_status(state)

    st.divider()

    render_dataset_overview(state)

    st.divider()

    render_current_best(state)

    st.divider()

    render_experiments(state)

    st.divider()

    render_critic(state)

    st.divider()

    render_final_evaluation(state)

    st.divider()

    render_dataset_details = st.checkbox(
        "Show detailed research information",
        value=False,
    )

    if render_dataset_details:
        render_feature_analysis(state)
        render_research_plan(state)
        render_experiment_history(state)
        render_critic_feedback(state)

    st.divider()

    render_final_report(state)


if __name__ == "__main__":
    main()