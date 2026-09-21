import datetime
import json
import os


def save_run_snapshot(state: dict, output_dir: str = "runs") -> str:
    """Dump the full state dict to a timestamped JSON file. Safe
    to call repeatedly during a run (e.g. after every experiment)
    so a crash doesn't lose everything gathered so far — just call
    it again with the latest state and it writes a new snapshot."""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    path = os.path.join(output_dir, f"run_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

    return path
