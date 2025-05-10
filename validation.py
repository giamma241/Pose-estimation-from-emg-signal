import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

def cross_validate_NN(nn_regressor, X_folds, Y_folds, metric_fns, n_folds=4, verbose=0):
    """_summary_

    Args:
        nn_regressor (_type_): _description_
        X_folds (_type_): _description_
        Y_folds (_type_): _description_
        metric_fns (_type_): _description_
        n_folds (int, optional): _description_. Defaults to 4.
        verbose (int, optional): _description_. Defaults to 0.
    """
    results = {}

    for fold in range(n_folds):
        if verbose == 3:
            print(f'FOLD {fold+1}/{n_folds}')
            
            fig = plt.figure()
            plt.title(f'Average batch losses per epoch - fold {fold+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            red_handle  = Line2D([0], [0], color='red',  marker='o', linestyle='-')
            blue_handle = Line2D([0], [0], color='blue', marker='s', linestyle='-')

            plt.legend(
                handles=[red_handle, blue_handle],
                labels=['Validation', 'Training'],
                title="Groups"
            )

        train_idx = list(range(n_folds))
        train_idx.remove(fold)
        val_idx = fold

        X_train = X_folds[train_idx].reshape(-1, *X_folds.shape[2:])
        Y_train = Y_folds[train_idx].reshape(-1, *Y_folds.shape[2:])
        X_val = X_folds[val_idx]
        Y_val = Y_folds[val_idx]

        if verbose <= 2:
            nn_regressor.fit(X_train, Y_train)
        else:
            train_losses, val_losses = nn_regressor.fit_with_validation(X_train, Y_train, X_val, Y_val)
            epochs = [i+1 for i in range(len(train_losses))]
            plt.plot(epochs, val_losses, marker='o', color = 'r')
            plt.plot(epochs, train_losses, marker='s', color = 'b')

        Y_train_pred = nn_regressor.predict(X_train)
        Y_val_pred = nn_regressor.predict(X_val)

        results[fold] = {}
        for name, fn in metric_fns.items():
            results[fold][f"train_{name}"] = fn(Y_train_pred, Y_train)
            results[fold][f"val_{name}"] = fn(Y_val_pred, Y_val)

        if verbose == 2:
            print(f"\nFold {fold + 1}")
            for name in metric_fns:
                print(
                    f"{name}: train={results[fold][f'train_{name}']:.4f}, val={results[fold][f'val_{name}']:.4f}"
                )
        
        if verbose == 3:
            plt.show()
    
    for name in metric_fns:
        # collect only the fold‐level metrics by indexing through the integer fold IDs
        train_vals = [results[fold][f"train_{name}"] for fold in range(n_folds)]
        val_vals   = [results[fold][f"val_{name}"]   for fold in range(n_folds)]

        results[f"avg_train_{name}"] = np.mean(train_vals)
        results[f"avg_val_{name}"]   = np.mean(val_vals)

    if verbose >= 1:
        print("\nAverage Scores across folds:")
        for name, fn in metric_fns.items():
            print(f'{name}: train={results[f"avg_train_{name}"]:.4f}, val={results[f"avg_val_{name}"]:.4f}')
    
    return results



def cross_validate_pipeline(pipeline, X_folds, Y_folds, metric_fns, n_folds=4, verbose=0):
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

        if verbose == 2:
            print(f"\nFold {fold + 1}")
            for name in metric_fns:
                print(
                    f"{name}: train={results[fold][f'train_{name}']:.4f}, val={results[fold][f'val_{name}']:.4f}"
                )
    
    for name in metric_fns:
        # collect only the fold‐level metrics by indexing through the integer fold IDs
        train_vals = [results[fold][f"train_{name}"] for fold in range(n_folds)]
        val_vals   = [results[fold][f"val_{name}"]   for fold in range(n_folds)]

        results[f"avg_train_{name}"] = np.mean(train_vals)
        results[f"avg_val_{name}"]   = np.mean(val_vals)

    if verbose >= 1:
        print("\nAverage Scores across folds:")
        for name, fn in metric_fns.items():
            print(f'{name}: train={results[f"avg_train_{name}"]:.4f}, val={results[f"avg_val_{name}"]:.4f}')
    
    return results


def parameter_selection(pipeline, param_grid, X_folds, Y_folds, metric_fns):
    """
    Performs parameter selection by modifying pipeline parameters and validating each configuration.

    Args:
        pipeline (Pipeline): a scikit-learn pipeline
        param_grid (dict): dictionary like {'regressor__alpha': [0.01, 0.1, 1]}
        X_folds (np.ndarray): training folds of shape (sessions, ...)
        Y_folds (np.ndarray): labels of shape (sessions, ...)
        metric_fns (dict): dictionary of scoring functions

    Returns:
        list of dicts: each entry contains 'params' and cross-val scores
    """

    all_results = []

    for params in ParameterGrid(param_grid):
        # Clone the pipeline and set parameters
        pipeline.set_params(**params)

        print(f"\nTesting parameters: {params}")
        result = cross_validate_pipeline(pipeline, X_folds, Y_folds, metric_fns, verbose=1)

        # Collect results
        result_summary = {}
        result_summary['params'] = params
        for name in metric_fns:
            result_summary[f'avg_train_{name}'] = result[f'avg_train_{name}']
            result_summary[f'avg_val_{name}'] = result[f'avg_val_{name}']

        all_results.append(result_summary)

    return all_results
