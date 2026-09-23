import sqlite3

from langgraph.graph import StateGraph, START, END

from app.state import ResearchState
from app.supervisor import supervisor
from app.config import settings
from app.logging_config import get_logger

from app.agents.data_agent import data_agent
from app.agents.feature_agent import feature_agent
from app.agents.research_agent import research_agent
from app.agents.split_agent import split_agent
from app.agents.experiment_agent import experiment_agent
from app.agents.evaluation_agent import evaluation_agent
from app.agents.critic_agent import critic_agent
from app.agents.feature_engineering_agent import (
    feature_engineering_agent
)
from app.agents.final_report_agent import final_report_agent


logger = get_logger(__name__)


def _get_checkpointer():
    """Create an optional SQLite checkpointer."""

    if not settings.ENABLE_CHECKPOINTING:
        return None

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning(
            "ENABLE_CHECKPOINTING is true but "
            "langgraph-checkpoint-sqlite is not installed. "
            "Continuing without checkpointing."
        )
        return None

    try:
        conn = sqlite3.connect(
            settings.CHECKPOINT_DB,
            check_same_thread=False
        )

        return SqliteSaver(conn)

    except Exception as error:
        logger.warning(
            f"Failed to initialize SQLite checkpointer: {error}. "
            "Continuing without checkpointing."
        )
        return None


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("data_agent", data_agent)
    graph.add_node("feature_agent", feature_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("split_agent", split_agent)
    graph.add_node("experiment_agent", experiment_agent)
    graph.add_node("evaluation_agent", evaluation_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node(
        "feature_engineering_agent",
        feature_engineering_agent
    )
    graph.add_node("supervisor", supervisor)
    graph.add_node("final_report_agent", final_report_agent)

    graph.add_edge(
        START,
        "data_agent"
    )

    graph.add_edge(
        "data_agent",
        "feature_agent"
    )

    graph.add_edge(
        "feature_agent",
        "research_agent"
    )

    graph.add_edge(
        "research_agent",
        "split_agent"
    )

    graph.add_edge(
        "split_agent",
        "experiment_agent"
    )

    graph.add_edge(
        "experiment_agent",
        "evaluation_agent"
    )

    graph.add_edge(
        "evaluation_agent",
        "critic_agent"
    )

    graph.add_edge(
        "critic_agent",
        "supervisor"
    )

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get(
            "next_agent",
            "finish"
        ),
        {
            "experiment_agent": "experiment_agent",
            "feature_engineering_agent": (
                "feature_engineering_agent"
            ),
            "finish": "final_report_agent",
        }
    )

    graph.add_conditional_edges(
        "feature_engineering_agent",
        lambda state: (
            "experiment_agent"
            if state.get(
                "feature_proposal",
                {}
            ).get("decision") == "propose"
            else "finish"
        ),
        {
            "experiment_agent": "experiment_agent",
            "finish": "final_report_agent",
        }
    )

    graph.add_edge(
        "final_report_agent",
        END
    )

    checkpointer = _get_checkpointer()

    if checkpointer is not None:
        return graph.compile(
            checkpointer=checkpointer
        )

    return graph.compile()