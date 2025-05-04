import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class TimeWindowTransformer:
    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def transform(self, X):
        """
        Takes in input a multichannel signal array of shape and returns time-windowed segments using sliding windows, producing shape

        Args:
            X (np.ndarray): input signal array (..., n_channels, n_times)

        Returns:
            np.ndarray: array of extracted time windows (..., n_windows, n_channels, size)
        """
        windows = sliding_window_view(X, window_shape=self.size, axis=-1)
        windows = windows[..., :: self.step, :]
        windows = np.moveaxis(windows, -2, -3)  # (..., n_windows, n_channels, size)
        return windows


class LabelWindowExtractor:
    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def transform(self, Y):
        """
        Takes in input a label array of shape and returns labels sampled at the end of each time window

        Args:
            Y (np.ndarray): array of time series labels (..., n_labels, n_times)

        Returns:
            np.ndarray: extracted window-aligned labels (..., n_windows, n_labels)
        """
        label_indices = np.arange(self.size - 1, Y.shape[-1], self.step)
        labels = Y[..., label_indices]
        labels = np.moveaxis(labels, -1, -2)  # (..., n_windows, n_labels)
        return labels


class TimeDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Takes in input a signal array of shape (..., n_times) or (sessions, windows, channels, n_times) and
    returns time-domain features computed over the last axis (time), resulting in (..., n_features)

    Feature list:
    [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT]

    Args:
        sigma_mpr (float, optional): threshold for Myopulse Percentage Rate. Defaults to 1.0.

    Returns:
        np.ndarray: the array of extracted features
    """

    def __init__(self, sigma_mpr=1.0):
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        original_shape = X.shape

        if X.ndim == 4:
            X = X.reshape(-1, X.shape[-1])

        MAV = np.mean(np.abs(X), axis=-1)  # Mean Absolute Value
        RMS = np.sqrt(np.mean(X**2, axis=-1))  # Root Mean Square
        VAR = np.var(X, ddof=1, axis=-1)  # Variance
        STD = np.std(X, ddof=1, axis=-1)  # Standard Deviation
        ZC = np.sum(X[..., :-1] * X[..., 1:] < 0, axis=-1)  # Zero crossing count
        MPR = np.mean(np.abs(X) > self.sigma_mpr, axis=-1)  # Myopulse percentage rate
        MAA = np.max(np.abs(X), axis=-1)  # Maximum absolute amplitude
        WL = np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)  # Waveform length

        SSC = np.sum(
            (-(X[..., 2:] - X[..., 1:-1]) * (X[..., 1:-1] - X[..., :-2]) > 0), axis=-1
        )  # slope sign changes
        WA = np.sum(
            np.abs(np.diff(X, axis=-1)) - STD[..., None] > 0, axis=-1
        )  # Wilson amplitude
        MFL = np.log(
            np.sqrt(np.sum(np.diff(X, axis=-1) ** 2, axis=-1))
        )  # Maximum fractal length
        KRT = stats.kurtosis(X, axis=-1, bias=False)  # Kurtosis

        features = np.stack(
            [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT], axis=-1
        )

        if len(original_shape) == 4:
            return features.reshape(
                original_shape[0], original_shape[1], original_shape[2], -1
            )
        else:
            return features

    def set_output(self, *, transform=None):
        return super().set_output(transform=transform)
