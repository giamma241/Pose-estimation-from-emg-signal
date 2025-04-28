import json
from pathlib import Path


def save_experiment_log(log, path="logs/experiment_log.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(log)

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
