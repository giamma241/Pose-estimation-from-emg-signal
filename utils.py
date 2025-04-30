import json
from pathlib import Path

import pandas as pd


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


def save_predictions(predictions, targets, path):
    df = pd.DataFrame(
        {
            **{f"pred_{i}": predictions[:, i] for i in range(predictions.shape[1])},
            **{f"true_{i}": targets[:, i] for i in range(targets.shape[1])},
        }
    )
    df.to_csv(path, index=False)


import numpy as np


class ExperimentSelector:
    def __init__(self, log_path):
        import json
        from pathlib import Path

        self.path = Path(log_path)
        with open(self.path) as f:
            self.logs = json.load(f)
        self.metrics = self._compute_metrics()

    def _compute_metrics(self):
        results = []

        for exp in self.logs:
            folds = exp.get("folds", [])
            rmse_curves = [f["metrics"].get("val_rmses", []) for f in folds]
            final_rmses = [c[-1] for c in rmse_curves if c]
            start_rmses = [c[0] for c in rmse_curves if len(c) > 1]
            lowest_rmses = [min(c) for c in rmse_curves if c]
            lowest_points = [min(c) for c in rmse_curves if c]

            if not final_rmses or not start_rmses:
                continue

            avg_final = np.mean(final_rmses)
            var_final = np.var(final_rmses)
            convergence = np.mean([s - f for s, f in zip(start_rmses, final_rmses)])
            avg_lowest = np.mean(lowest_rmses)

            results.append(
                {
                    "architecture": exp["architecture"],
                    "avg_final_rmse": avg_final,
                    "rmse_var": var_final,
                    "convergence": convergence,
                    "avg_lowest_rmse": avg_lowest,
                    "fold_rmses": final_rmses,
                    "fold_curves": rmse_curves,
                }
            )

        return results

    def select(
        self,
        sort_by="final_avg",
        top_n=5,
        max_variance=None,
        min_convergence=None,
        return_full=False,
    ):
        key_funcs = {
            "final_avg": lambda x: x["avg_final_rmse"],
            "lowest_point": lambda x: x["avg_lowest_rmse"],
            "variance": lambda x: x["rmse_var"],
            "convergence": lambda x: -x["convergence"],
            "potential": lambda x: x["avg_final_rmse"] - x["convergence"],
            "bias": lambda x: np.mean(x["fold_rmses"])
            - np.mean([c[0] for c in x["fold_curves"] if c]),
        }

        if sort_by not in key_funcs:
            raise ValueError(f"Unsupported sort_by: {sort_by}")

        filtered = self.metrics

        if max_variance is not None:
            filtered = [m for m in filtered if m["rmse_var"] <= max_variance]
        if min_convergence is not None:
            filtered = [m for m in filtered if m["convergence"] >= min_convergence]

        ranked = sorted(filtered, key=key_funcs[sort_by])[:top_n]

        if return_full:

            def find_full(arch):
                return next(exp for exp in self.logs if exp["architecture"] == arch)

            return [find_full(m["architecture"]) for m in ranked]
        else:
            slimmed = []
            for m in ranked:
                slimmed.append(
                    {
                        "architecture": m["architecture"],
                        "avg_final_rmse": m["avg_final_rmse"],
                        "rmse_var": m["rmse_var"],
                        "convergence": m["convergence"],
                        "avg_lowest_rmse": m.get("avg_lowest_rmse"),
                    }
                )
            return slimmed
