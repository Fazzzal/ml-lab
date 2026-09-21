import datetime
import logging
import sys
from pathlib import Path

from app.config import settings

_LOG_DIR = Path(settings.LOGS_DIR)
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TIMESTAMP = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

LOG_FILE = _LOG_DIR / f"run_{_RUN_TIMESTAMP}.log"

# When stdout isn't a real terminal (common with IDE run
# configurations, piped output, etc.), Python can block-buffer it
# instead of flushing line by line — console output then only
# appears in chunks, or not at all until the process exits. Force
# line buffering so log lines show up immediately regardless of
# how this script is launched.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    # Not all stdout wrappers support reconfigure (e.g. some
    # captured/piped streams); harmless to skip.
    pass


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record, belt-and-
    braces alongside the line-buffering above."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both stdout and a single
    per-process log file, shared across every agent/tool module
    that calls this."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        stream_handler = _FlushingStreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            LOG_FILE, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False

    return logger