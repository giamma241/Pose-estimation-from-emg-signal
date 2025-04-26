import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import sliding_window_view


def load_data(x_path, y_path):
    """Load X and Y datasets from .npy files."""
    X = np.load(x_path)
    Y = np.load(y_path)
    return X, Y


def split_sessions(X, Y, train_sessions=[0, 1, 2, 3], val_session=4):
    """Split datasets into training and validation sessions."""
    X_train = X[train_sessions]
    Y_train = Y[train_sessions]
    X_val = X[val_session]
    Y_val = Y[val_session]
    return X_train, Y_train, X_val, Y_val


def create_time_windows(X, size=500, step=100):
    """Create sliding time windows from the input signals."""
    windows = sliding_window_view(X, window_shape=size, axis=-1)
    windows = windows[..., ::step, :]
    windows = windows.transpose(0, 2, 1, 3)
    return windows


def create_labels(Y, size=500, step=100):
    """Extract labels corresponding to the end of each window."""
    label_indices = np.arange(size - 1, Y.shape[-1], step)
    labels = Y[..., label_indices]
    labels = labels.transpose(0, 2, 1)
    return labels


# Define feature functions
def MAV(x):
    """Mean Absolute Value."""
    return np.mean(np.abs(x))


def RMS(x):
    """Root Mean Square."""
    return np.sqrt(np.mean(x**2))


def VAR(x):
    """Variance."""
    return np.var(x, ddof=1)


def STD(x):
    """Standard Deviation."""
    return np.std(x, ddof=1)


def ZC(x):
    """Zero Crossing count."""
    return np.sum(x[:-1] * x[1:] < 0)


def MPR(x, sigma=0.3):
    """Myopulse Percentage Rate."""
    return np.mean(np.abs(x) > sigma)


def MAA(x):
    """Maximum Absolute Amplitude."""
    return np.max(np.abs(x))


def WL(x):
    """Waveform Length."""
    return np.sum(np.abs(np.diff(x)))


def SSC(x):
    """Slope Sign Changes."""
    return np.sum(-(x[2:] - x[1:-1]) * (x[1:-1] - x[:-2]) > 0)


def WA(x):
    """Wilson Amplitude."""
    return np.sum(np.abs(np.diff(x)) - STD(x) > 0)


def MFL(x):
    """Maximum Fractal Length."""
    return np.log(np.sqrt(np.sum(np.diff(x) ** 2)))


def KRT(x):
    """Kurtosis."""
    return sp.stats.kurtosis(x, bias=False)


time_domain_features = {
    "MAV": (MAV, {}),
    "RMS": (RMS, {}),
    "VAR": (VAR, {}),
    "STD": (STD, {}),
    "ZC": (ZC, {}),
    "MPR": (MPR, {"sigma": 0.3}),
    "MAA": (MAA, {}),
    "WL": (WL, {}),
    "SSC": (SSC, {}),
    "WA": (WA, {}),
    "MFL": (MFL, {}),
    "KRT": (KRT, {}),
}


def extract_time_domain_features(windows):
    """Extract time domain features from sliding windows."""
    sessions, windows_n, electrodes, size = windows.shape
    n_features = len(time_domain_features)
    feature_array = np.zeros((sessions, windows_n, electrodes, n_features))

    for idx, (key, (func, kwargs)) in enumerate(time_domain_features.items()):
        feature_array[:, :, :, idx] = np.apply_along_axis(func, -1, windows, **kwargs)

    return feature_array


def preprocess_dataset(x_path, y_path, mode="raw", window_size=500, step_size=100):
    """
    Full preprocessing pipeline: load data, split sessions, create windows, extract features.

    Args:
        x_path: Path to X dataset file.
        y_path: Path to Y dataset file.
        mode: Output mode ('raw', 'windows', or 'features').
        window_size: Size of each sliding window.
        step_size: Step size between windows.

    Returns:
        Preprocessed datasets according to selected mode.
    """
    X, Y = load_data(x_path, y_path)
    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw = split_sessions(X, Y)

    if mode == "raw":
        return X_train_raw, Y_train_raw, X_val_raw, Y_val_raw

    X_train_windows = create_time_windows(X_train_raw, size=window_size, step=step_size)
    Y_train = create_labels(Y_train_raw, size=window_size, step=step_size)
    X_val_windows = create_time_windows(
        np.expand_dims(X_val_raw, axis=0), size=window_size, step=step_size
    )
    Y_val = create_labels(
        np.expand_dims(Y_val_raw, axis=0), size=window_size, step=step_size
    )

    if mode == "windows":
        return X_train_windows, Y_train, X_val_windows, Y_val

    if mode == "features":
        X_train_features = extract_time_domain_features(X_train_windows)
        X_val_features = extract_time_domain_features(X_val_windows)
        return X_train_features, Y_train, X_val_features, Y_val

    raise ValueError("Invalid mode. Choose from 'raw', 'windows', or 'features'.")
