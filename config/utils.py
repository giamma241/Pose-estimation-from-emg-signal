import json
from pathlib import Path

import pandas as pd


def save_to_csv(Y, fname):
    """
    input: a numpy array of shape (N, 51)
    """
    Z = Y.reshape(-1, 17, 3)
    with open(fname, 'w') as f:
        for row in Z:
            triplets = [f'{row[i,0]},{row[i,1]},{row[i,2]}' for i in range(17)]
            f.write(';'.join(triplets) + ';\n')


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


import random

import numpy as np


class ArchitectureSampler:
    def __init__(
        self,
        widths=(128, 256, 512, 768, 1024, 1536),
        depths=(2, 3, 4, 5),
        shapes=("increasing", "decreasing", "symmetric", "flat"),
        limit_per_shape=10,
        dropout_levels=(0.0, 0.05, 0.1),
        max_fc_depth=None,
        max_conv_depth=None,
        conv_templates=None,
        output_dim=51,
        seed=None,
        verbose=False,
    ):
        self.widths = widths
        self.depths = depths
        self.shapes = shapes
        self.limit = limit_per_shape
        self.dropout_levels = dropout_levels
        self.max_fc_depth = max_fc_depth
        self.max_conv_depth = max_conv_depth
        self.output_dim = output_dim
        self.verbose = verbose
        self.conv_templates = conv_templates
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        self._configs = []

    def _generate_shape(self, shape, depth):
        if shape == "increasing":
            return sorted(random.sample(self.widths, depth))
        elif shape == "decreasing":
            return sorted(random.sample(self.widths, depth), reverse=True)
        elif shape == "symmetric":
            half = sorted(random.sample(self.widths, depth // 2))
            return (
                half + half[::-1]
                if depth % 2 == 0
                else half + [random.choice(self.widths)] + half[::-1]
            )
        elif shape == "flat":
            w = random.choice(self.widths)
            return [w] * depth
        else:
            raise ValueError(f"Unknown shape type: {shape}")

    def generate_fc_templates(self):
        fc_templates = {}
        for shape in self.shapes:
            for depth in self.depths:
                key = f"{shape}_d{depth}"
                fc_templates[key] = [
                    self._generate_shape(shape, depth) for _ in range(self.limit)
                ]
        return fc_templates

    def generate_configs(self):
        fc_templates = self.generate_fc_templates()

        # fallback if no conv templates provided
        conv_templates = self.conv_templates or {
            "shallow_wide": [[(64, 5, 1)], [(128, 5, 1), (128, 3, 1)]],
            "deep_narrow": [[(32, 3, 1)] * 4, [(64, 3, 1)] * 5],
            "expanding": [[(32, 5, 1), (64, 3, 1), (128, 3, 1)]],
            "bottleneck": [[(128, 5, 1), (64, 3, 1), (32, 3, 1)]],
            "oscillating": [[(64, 5, 1), (128, 3, 1), (64, 3, 1)]],
        }

        configs = []

        for conv_name, conv_list in conv_templates.items():
            for fc_name, fc_list in fc_templates.items():
                for conv_cfg in conv_list:
                    if self.max_conv_depth and len(conv_cfg) > self.max_conv_depth:
                        continue
                    for fc_cfg in fc_list:
                        if self.max_fc_depth and len(fc_cfg) > self.max_fc_depth:
                            continue
                        for d in self.dropout_levels:
                            config = {
                                "conv_layers_config": list(conv_cfg),
                                "fc_layers_config": list(fc_cfg),
                                "conv_dropouts": [d] * len(conv_cfg),
                                "fc_dropouts": [d] * len(fc_cfg),
                                "output_dim": self.output_dim,
                                "verbose": self.verbose,
                                "conv_family": conv_name,
                                "fc_family": fc_name,
                                "dropout_level": d,
                            }
                            configs.append(config)

        self._configs = configs
        return configs

    def get_all_configs(self):
        return self._configs

    def get_model_configs(self):
        keys = [
            "conv_layers_config",
            "fc_layers_config",
            "conv_dropouts",
            "fc_dropouts",
            "output_dim",
            "verbose",
        ]
        return [{k: c[k] for k in keys} for c in self._configs]

    def shuffle_configs(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._configs)


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
