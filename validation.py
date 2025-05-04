import numpy as np


def RMSE(y_pred, y_val):
    """
    Computes the root mean squared error of y_pred with respect to y_val.
    Assumes a shape (N, 51)
    """
    return (np.sum((y_val - y_pred) ** 2) / np.prod(y_val.shape)) ** 0.5


def NMSE(y_pred, y_val):
    """
    Computes the normalized mean squared error of y_pred with respect to y_val.
    Assumes a shape (N, 51)
    """
    num = np.sum((y_val - y_pred) ** 2)
    den = np.sum((y_val - np.mean(y_val, axis=0)) ** 2)
    return num / den


def cross_validate_pipeline(pipeline, X, Y, metric_fns, n_folds=4):
    """
    Performs leave-one-session-out cross-validation for a pipeline.

    Args:
        pipeline: a sklearn-compatible pipeline
        X (np.ndarray): input features of shape (sessions, windows, ...)
        Y (np.ndarray): target labels of shape (sessions, windows, ...)
        metric_fns (dict): dictionary of metric functions like {'RMSE': rmse_fn, 'NMSE': nmse_fn}
        n_folds (int): number of folds (typically 4 sessions for training)

    Returns:
        dict: results per fold + mean summary
    """
    results = {}

    for fold in range(n_folds):
        train_idx = list(range(n_folds))
        train_idx.remove(fold)
        val_idx = fold

        X_train = X[train_idx].reshape(-1, *X.shape[2:])
        Y_train = Y[train_idx].reshape(-1, *Y.shape[2:])
        X_val = X[val_idx]
        Y_val = Y[val_idx]

        pipeline.fit(X_train, Y_train)
        Y_train_pred = pipeline.predict(X_train)
        Y_val_pred = pipeline.predict(X_val)

        results[fold] = {}
        for name, fn in metric_fns.items():
            results[fold][f"train_{name}"] = fn(Y_train_pred, Y_train)
            results[fold][f"val_{name}"] = fn(Y_val_pred, Y_val)

        print(f"\nFold {fold + 1}")
        for name in metric_fns:
            print(
                f"{name}: train={results[fold][f'train_{name}']:.4f}, val={results[fold][f'val_{name}']:.4f}"
            )

    # Compute mean scores
    mean_scores = {
        f"mean_val_{k.split('_')[1]}": np.mean(
            [fold[k] for fold in results.values() if k.startswith("val_")]
        )
        for k in results[0]
    }
    print("\nMean Validation Scores:")
    for k, v in mean_scores.items():
        print(f"{k}: {v:.4f}")

    return {"folds": results, "summary": mean_scores}
