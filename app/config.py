import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL", "openai/gpt-oss-120b"
    )

    # Supervisor loop safety net
    MAX_ITERATIONS: int = int(
        os.getenv("MAX_ITERATIONS", "15")
    )

    # Data splitting. TEST is held out and touched exactly once,
    # at the very end, by final_report_agent. VAL is what the
    # critic loop compares experiments against.
    TEST_SIZE: float = float(
        os.getenv("TEST_SIZE", "0.2")
    )
    VAL_SIZE: float = float(
        os.getenv("VAL_SIZE", "0.2")
    )
    RANDOM_STATE: int = int(
        os.getenv("RANDOM_STATE", "42")
    )

    # Cross-validation folds computed on the training split, in
    # addition to the single validation-set score. 0 disables it.
    CV_FOLDS: int = int(
        os.getenv("CV_FOLDS", "5")
    )

    # LLM call reliability. Without an explicit timeout, a slow or
    # stalled Groq response hangs the whole run with no way to
    # tell it apart from normal work — better to fail fast and
    # let the calling agent decide what to do.
    LLM_TIMEOUT_SECONDS: int = int(
        os.getenv("LLM_TIMEOUT_SECONDS", "45")
    )
    LLM_MAX_RETRIES: int = int(
        os.getenv("LLM_MAX_RETRIES", "2")
    )

    # Default dataset, overridable per-run without editing code.
    DATASET_PATH: str = os.getenv(
        "DATASET_PATH", "data/Telco-Customer-Churn.csv"
    )
    TARGET_COLUMN: str = os.getenv(
        "TARGET_COLUMN", "Churn"
    )

    # Persistence
    ENABLE_CHECKPOINTING: bool = _get_bool(
        "ENABLE_CHECKPOINTING", False
    )
    CHECKPOINT_DB: str = os.getenv(
        "CHECKPOINT_DB", "checkpoints.db"
    )
    RUN_ID: str = os.getenv(
        "RUN_ID", "default-run"
    )
    RUNS_DIR: str = os.getenv(
        "RUNS_DIR", "runs"
    )
    LOGS_DIR: str = os.getenv(
        "LOGS_DIR", "logs"
    )


settings = Settings()