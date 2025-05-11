import matplotlib.pyplot as plt
import numpy as np
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


def cross_validate_pipeline(
    pipeline, X_folds, Y_folds, metric_fns, n_folds=4, verbose=0
):
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
        val_vals = [results[fold][f"val_{name}"] for fold in range(n_folds)]

        results[f"avg_train_{name}"] = np.mean(train_vals)
        results[f"avg_val_{name}"] = np.mean(val_vals)

    if verbose >= 1:
        print("\nAverage Scores across folds:")
        for name, fn in metric_fns.items():
            print(
                f"{name}: train={results[f'avg_train_{name}']:.4f}, val={results[f'avg_val_{name}']:.4f}"
            )

    return results


def mutual_info_corr(x, y):
    """
    Approximates the mutual information between two 1D variables x and y
    using their Pearson correlation coefficient.

    This approximation assumes a Gaussian relationship and computes:
        MI(x, y) ≈ -0.5 * log(1 - ρ²) where ρ is the Pearson correlation coefficient between x and y.

    Args:
        x (np.ndarray): 1D array of values
        y (np.ndarray): 1D array of values (same length as x)

    Returns:
        float: estimated mutual information between x and y
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return 0.0
    if abs(c) == 1:
        c = 0.999999
    return -0.5 * np.log(1 - c**2)


def compute_mi_vector(X_raw, Y_sessions, sigma_mpr=0.3):
    """
    Computes average mutual information between each extracted feature and all targets.
    Feature extraction (TimeDomainTransformer) is applied internally.

    Args:
        X_raw (np.ndarray): raw EMG of shape (n_sessions, n_windows, n_channels, window_size)
        Y_sessions (np.ndarray): shape (n_sessions, n_windows, n_outputs)
        sigma_mpr (float): threshold for MPR feature

    Returns:
        X_df: DataFrame of features (flattened across sessions)
        Y_all: Flattened output labels
        mi_scores: mutual information scores per feature
        X_sessions: (n_sessions, n_windows, n_features) for later use
    """
    n_sessions, n_windows, n_channels, window_size = X_raw.shape
    td_transformer = TimeDomainTransformer(sigma_mpr=sigma_mpr)
    X_feat = td_transformer.transform(X_raw)  # (sessions, windows, channels, 12)
    X_sessions = X_feat.reshape(n_sessions, n_windows, -1)
    Y_all = Y_sessions.reshape(-1, Y_sessions.shape[-1])
    X_df = pd.DataFrame(X_sessions.reshape(-1, X_sessions.shape[-1]))

    mis = []
    for col in X_df.columns:
        mi_vals = [
            mutual_info_corr(X_df[col].values, Y_all[:, j])
            for j in range(Y_all.shape[1])
        ]
        mis.append(np.mean(mi_vals))

    return X_df, Y_all, np.array(mis), X_sessions


def greedy_mrmr_selection(X_df, mi_scores):
    """
    Performs greedy mRMR selection.
    Args:
        X_df: feature matrix (DataFrame)
        mi_scores: precomputed relevance scores (1D array)
    Returns:
        selected: list of selected feature indices ordered by importance
    """
    selected = []
    candidates = list(range(X_df.shape[1]))

    for _ in range(len(candidates)):
        redundancy = np.zeros(len(candidates))
        if selected:
            for i, cidx in enumerate(candidates):
                cc_vals = []
                for sidx in selected:
                    cc = np.corrcoef(X_df.iloc[:, sidx], X_df.iloc[:, cidx])[0, 1]
                    if abs(cc) == 1:
                        cc = 0.999999
                    cc_vals.append(-0.5 * np.log(1 - cc**2))
                redundancy[i] = np.mean(cc_vals)

        mrmr_score = mi_scores[candidates] - redundancy
        best = candidates[np.argmax(mrmr_score)]
        selected.append(best)
        candidates.remove(best)

    return selected
