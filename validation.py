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


def cross_validate_pipeline(pipeline, X_folds, Y_folds, metric_fns, n_folds=4):
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

        X_train = X_folds[train_idx].reshape(-1, *X_folds.shape[2:])
        Y_train = Y_folds[train_idx].reshape(-1, *Y_folds.shape[2:])
        X_val = X_folds[val_idx]
        Y_val = Y_folds[val_idx]

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
        
    avg_train_RMSE = np.mean([dic['train_RMSE'] for dic in results.values()])
    avg_val_RMSE = np.mean([dic['val_RMSE'] for dic in results.values()])
    avg_train_NMSE = np.mean([dic['train_NMSE'] for dic in results.values()])
    avg_val_NMSE = np.mean([dic['val_NMSE'] for dic in results.values()])

    print("\nMean Validation Scores:")
    print(f'RMSE: train={avg_train_RMSE:.4f}, val={avg_val_RMSE:.4f}')
    print(f'NMSE: train={avg_train_NMSE:.4f}, val={avg_val_NMSE:.4f}')
    
    return results