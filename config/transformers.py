import nolds
import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import butter, decimate, filtfilt, iirnotch, resample, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

from config.validation import mutual_info_corr

# Time Windows


class TimeWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that segments multichannel signal arrays into time windows
    using sliding windows. Takes an input signal array and divides it into overlapping
    time windows.

    Attributes:
        size (int): The size of each time window (number of time points).
        step (int): The step size of the sliding window (number of time points
                    to move the window forward).
    """

    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

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


class LabelWindowExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts labels corresponding to time windows from
    a label array. Takes an input label array and samples the labels at the end
    of each time window.

    Attributes:
        size (int): The size of each time window (number of time points),
                    which determines where to sample the label.
        step (int): The step size used to create the time windows, which
                    determines the interval between label samples.
    """

    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

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


# Time domain


class TimeDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts a set of standard time-domain features
    from the last axis of a NumPy array of shape (..., n_times),
    returning an array of shape (..., N).

    Attributes:
        sigma_mpr (float, optional): Value of sigma for the Myopulse Percentage Rate
                                     (MPR) calculation.  Defaults to 1.0.
    """

    def __init__(self, sigma_mpr=0.3):
        """
        Parameters:
            sigma_mpr (float, optional):    value of sigma for the Myopulse Percentage Rate.
                                            Defaults to 0.3.
        """
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

    def transform(self, X):
        """
        Extracts time-domain features from the input signal array.

        Args:
            X (np.ndarray): Input signal array of shape (..., n_times).

        Returns:
            np.ndarray: Array of EMG time-domain features of shape (..., n_features),
                        where n_features corresponds to the number of extracted
                        features. The features are:
                        [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT]
        """
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

        ans = np.stack(
            [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT], axis=-1
        )
        return ans.reshape(*ans.shape[:-2], -1)

    def set_output(self, *, transform=None):
        # Here for compatibility
        return super().set_output(transform=transform)


class ExtendedTimeDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts an extended set of time-domain features.
    """

    def __init__(self, sigma_mpr=1.0):
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        MAV = np.mean(np.abs(X), axis=-1)
        RMS = np.sqrt(np.mean(X**2, axis=-1))
        VAR = np.var(X, ddof=1, axis=-1)
        STD = np.std(X, ddof=1, axis=-1)
        ZC = np.sum(X[..., :-1] * X[..., 1:] < 0, axis=-1)
        MPR = np.mean(np.abs(X) > self.sigma_mpr, axis=-1)
        MAA = np.max(np.abs(X), axis=-1)
        WL = np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)
        SSC = np.sum(
            (-(X[..., 2:] - X[..., 1:-1]) * (X[..., 1:-1] - X[..., :-2]) > 0), axis=-1
        )
        WA = np.sum(np.abs(np.diff(X, axis=-1)) - STD[..., None] > 0, axis=-1)
        MFL = np.log(
            np.sqrt(np.sum(np.diff(X, axis=-1) ** 2, axis=-1) + 1e-8)
        )  # Added small epsilon for numerical stability
        KRT = stats.kurtosis(X, axis=-1, bias=False)
        SKW = stats.skew(X, axis=-1, bias=False)
        MOM3 = stats.moment(X, moment=3, axis=-1)

        # Temporal Complexity (example using nolds - install it: pip install nolds)
        sampen = np.array(
            [nolds.sampen(sig)[0] if len(sig) > 2 else np.nan for sig in X]
        )  # Apply per channel
        sampen = np.nan_to_num(
            sampen.reshape(*X.shape[:-1], 1)
        )  # Handle NaN and reshape

        dfa = np.array(
            [nolds.dfa(sig) if len(sig) > 10 else np.nan for sig in X]
        )  # Apply per channel
        dfa = np.nan_to_num(dfa.reshape(*X.shape[:-1], 1))  # Handle NaN and reshape

        # Integrated EMG
        IEMG = np.sum(np.abs(X), axis=-1)

        ans = np.stack(
            [
                MAV,
                RMS,
                VAR,
                STD,
                ZC,
                MPR,
                MAA,
                WL,
                SSC,
                WA,
                MFL,
                KRT,
                SKW,
                MOM3,
                IEMG,
                sampen.squeeze(-1),
                dfa.squeeze(-1),
            ],
            axis=-1,
        )
        return ans.reshape(*ans.shape[:-2], -1)


# Feasture selection


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Selects features from the input array.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features) or
                           (n_sessions, n_windows, n_features).

        Returns:
            np.ndarray: Array of selected features, guaranteed to be 2D
                        with shape (n_samples, n_selected_features).
        """
        # Handle 3D shape (n_sessions, n_windows, n_features)
        if X.ndim == 3:
            n_sessions, n_windows, n_features = X.shape
            X = X.reshape(n_sessions * n_windows, n_features)
        elif X.ndim != 2:
            raise ValueError(f"Unsupported input shape: {X.shape}")

        # Defensive check
        if max(self.feature_indices) >= X.shape[1]:
            raise IndexError(
                f"Selected index {max(self.feature_indices)} "
                f"exceeds input feature dimension {X.shape[1]}"
            )

        return X[:, self.feature_indices]


def mutual_info_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return 0.0
    if abs(c) == 1:
        c = 0.999999
    return -0.5 * np.log(1 - c**2)


class TopKMRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k
        self.selected_indices = None

    def fit(self, X, y):
        import numpy as np
        import pandas as pd

        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        if y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        X_df = pd.DataFrame(X)
        y_mat = y

        relevance = []
        for col in X_df.columns:
            rel = np.mean(
                [
                    mutual_info_corr(X_df[col].values, y_mat[:, j])
                    for j in range(y_mat.shape[1])
                ]
            )
            relevance.append(rel)

        selected = []
        candidates = list(range(X_df.shape[1]))

        for _ in range(self.k):
            redundancy = np.zeros(len(candidates))
            if selected:
                for i, c in enumerate(candidates):
                    red = np.mean(
                        [
                            mutual_info_corr(X_df[c].values, X_df[s].values)
                            for s in selected
                        ]
                    )
                    redundancy[i] = red
            scores = np.array(relevance)[candidates] - redundancy
            best = candidates[np.argmax(scores)]
            selected.append(best)
            candidates.remove(best)

        self.selected_indices = selected
        return self

    def transform(self, X):
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        return X[:, self.selected_indices]


# Filter


class EmgFilterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a series of filters to EMG signals: resampling,
    notch filtering to remove power line interference, and bandpass filtering
    to isolate the relevant frequency components of the EMG signal. The
    filtering is applied independently to each channel of each sample in the
    input data.

    Attributes:
        original_fs (int): The original sampling frequency of the EMG signals.
        target_fs (int): The target sampling frequency after the initial upsampling.
        f0 (float): The center frequency of the notch filter (e.g., 50.0 Hz for EU power).
        bw (float): The bandwidth of the notch filter.
        low (float): The lower cutoff frequency of the bandpass filter.
        high (float): The upper cutoff frequency of the bandpass filter.
        order (int): The order of the Butterworth bandpass filter.
    """

    def __init__(
        self,
        original_fs=1024,
        target_fs=2048,
        f0=50.0,
        bw=5.0,
        low=30.0,
        high=500.0,
        order=4,
    ):
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.f0 = f0
        self.bw = bw
        self.low = low
        self.high = high
        self.order = order

    def fit(self, X, y=None):
        # No fitting required
        return self

    def transform(self, X):
        """
        Applies the filtering pipeline to the input EMG data.

        Args:
            X (np.ndarray): Input EMG data of shape (samples, channels, time).

        Returns:
            np.ndarray: Filtered EMG data of the same shape as the input.

        Raises:
            ValueError: If the input data does not have 3 dimensions.
        """
        # Apply filtering to each sample independently
        if X.ndim != 3:
            raise ValueError("Expected input of shape (samples, channels, time)")
        filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for ch in range(X.shape[1]):
                filtered[i, ch] = self._filter_channel(X[i, ch])
        return filtered

    def _filter_channel(self, x):
        # Upsample
        x_resampled = self._resample_signal(x, self.original_fs, self.target_fs)

        # Notch filter
        Q = self.f0 / self.bw
        b_notch, a_notch = iirnotch(self.f0, Q, self.target_fs)
        x_notched = filtfilt(b_notch, a_notch, x_resampled)

        # Bandpass filter
        sos = butter(
            self.order,
            [self.low, self.high],
            btype="bandpass",
            fs=self.target_fs,
            output="sos",
        )
        x_filtered = sosfiltfilt(sos, x_notched)

        # Downsample back
        return self._resample_signal(x_filtered, self.target_fs, self.original_fs)

    def _resample_signal(self, signal, original_fs, target_fs):
        ratio = original_fs / target_fs
        if ratio > 1:
            if not ratio.is_integer():
                raise ValueError("Use Method 2 for non-integer ratios")
            return decimate(signal, int(ratio), zero_phase=True)
        elif ratio < 1:
            num_samples = int(len(signal) * target_fs / original_fs)
            return resample(signal, num_samples)
        return signal


class EMGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, sampling_freq=1024, bandpass_range=(20, 450), order=4, rms_window_ms=100
    ):
        self.sampling_freq = sampling_freq
        self.bandpass_range = bandpass_range
        self.order = order
        self.rms_window_samples = int(rms_window_ms * sampling_freq / 1000)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_processed = np.zeros_like(X)
        for session in range(X.shape[0]):
            emg = X[session]
            emg_rectified = self._full_wave_rectification(emg)
            emg_filtered = self._butter_filter_signal(emg_rectified)
            emg_smoothed = self._rms_smoothing(emg_filtered)
            X_processed[session] = emg_smoothed
        return X_processed

    def _full_wave_rectification(self, emg_data):
        return np.abs(emg_data)

    def _butter_filter(self, cutoff_freq, btype):
        nyquist_freq = 0.5 * self.sampling_freq
        if isinstance(cutoff_freq, (list, tuple)):
            normal_cutoff = [f / nyquist_freq for f in cutoff_freq]
        else:
            normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(self.order, normal_cutoff, btype=btype, analog=False)
        return b, a

    def _butter_filter_signal(self, emg_data):
        emg_data_filtered = np.zeros_like(emg_data)
        for ch in range(emg_data.shape[0]):
            b, a = self._butter_filter(self.bandpass_range, btype="band")
            emg_data_filtered[ch, :] = filtfilt(b, a, emg_data[ch, :])
        return emg_data_filtered

    def _rms_smoothing(self, emg_data):
        smoothed = np.zeros_like(emg_data)
        half_win = self.rms_window_samples // 2
        for ch in range(emg_data.shape[0]):
            for t in range(half_win, emg_data.shape[1] - half_win):
                window = emg_data[ch, t - half_win : t + half_win]
                smoothed[ch, t] = np.sqrt(np.mean(window**2))
        return smoothed


class IEMGFilterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, integration_window_ms=100, sampling_freq=1024):
        self.integration_window_samples = int(
            (integration_window_ms / 1000) * sampling_freq
        )
        self.sampling_freq = sampling_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError("Expected input shape (samples, channels, time)")

        n_samples, n_channels, _ = X.shape
        output = []

        for i in range(n_samples):
            sample_out = []
            for ch in range(n_channels):
                signal = X[i, ch, :]

                # Rectify
                rectified = np.abs(signal)

                # Integrate
                kernel = (
                    np.ones(self.integration_window_samples)
                    / self.integration_window_samples
                )
                iemg = np.convolve(rectified, kernel, mode="same")

                # Normalize
                iemg_min = np.min(iemg)
                iemg_max = np.max(iemg)
                iemg_norm = (iemg - iemg_min) / (iemg_max - iemg_min + 1e-8)

                sample_out.append(iemg_norm)
            output.append(sample_out)

        return np.array(output)


class TimeWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that segments multichannel signal arrays into time windows
    using sliding windows. Takes an input signal array and divides it into overlapping
    time windows.

    Attributes:
        size (int): The size of each time window (number of time points).
        step (int): The step size of the sliding window (number of time points
                    to move the window forward).
    """

    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

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


class LabelWindowExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts labels corresponding to time windows from
    a label array. Takes an input label array and samples the labels at the end
    of each time window.

    Attributes:
        size (int): The size of each time window (number of time points),
                    which determines where to sample the label.
        step (int): The step size used to create the time windows, which
                    determines the interval between label samples.
    """

    def __init__(self, size=500, step=100):
        self.size = size
        self.step = step

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

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
    Transformer that extracts a set of standard time-domain features
    from the last axis of a NumPy array of shape (..., n_times),
    returning an array of shape (..., N).

    Attributes:
        sigma_mpr (float, optional): Value of sigma for the Myopulse Percentage Rate
                                     (MPR) calculation.  Defaults to 1.0.
    """

    def __init__(self, sigma_mpr=0.3):
        """
        Parameters:
            sigma_mpr (float, optional):    value of sigma for the Myopulse Percentage Rate.
                                            Defaults to 0.3.
        """
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

    def transform(self, X):
        """
        Extracts time-domain features from the input signal array.

        Args:
            X (np.ndarray): Input signal array of shape (..., n_times).

        Returns:
            np.ndarray: Array of EMG time-domain features of shape (..., n_features),
                        where n_features corresponds to the number of extracted
                        features. The features are:
                        [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT]
        """
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

        ans = np.stack(
            [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT], axis=-1
        )
        return ans.reshape(*ans.shape[:-2], -1)

    def set_output(self, *, transform=None):
        # Here for compatibility
        return super().set_output(transform=transform)


class ExtendedTimeDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts an extended set of time-domain features.
    """

    def __init__(self, sigma_mpr=1.0):
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        MAV = np.mean(np.abs(X), axis=-1)
        RMS = np.sqrt(np.mean(X**2, axis=-1))
        VAR = np.var(X, ddof=1, axis=-1)
        STD = np.std(X, ddof=1, axis=-1)
        ZC = np.sum(X[..., :-1] * X[..., 1:] < 0, axis=-1)
        MPR = np.mean(np.abs(X) > self.sigma_mpr, axis=-1)
        MAA = np.max(np.abs(X), axis=-1)
        WL = np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)
        SSC = np.sum(
            (-(X[..., 2:] - X[..., 1:-1]) * (X[..., 1:-1] - X[..., :-2]) > 0), axis=-1
        )
        WA = np.sum(np.abs(np.diff(X, axis=-1)) - STD[..., None] > 0, axis=-1)
        MFL = np.log(
            np.sqrt(np.sum(np.diff(X, axis=-1) ** 2, axis=-1) + 1e-8)
        )  # Added small epsilon for numerical stability
        KRT = stats.kurtosis(X, axis=-1, bias=False)
        SKW = stats.skew(X, axis=-1, bias=False)
        MOM3 = stats.moment(X, moment=3, axis=-1)

        # Temporal Complexity (example using nolds - install it: pip install nolds)
        sampen = np.array(
            [nolds.sampen(sig)[0] if len(sig) > 2 else np.nan for sig in X]
        )  # Apply per channel
        sampen = np.nan_to_num(
            sampen.reshape(*X.shape[:-1], 1)
        )  # Handle NaN and reshape

        dfa = np.array(
            [nolds.dfa(sig) if len(sig) > 10 else np.nan for sig in X]
        )  # Apply per channel
        dfa = np.nan_to_num(dfa.reshape(*X.shape[:-1], 1))  # Handle NaN and reshape

        # Integrated EMG
        IEMG = np.sum(np.abs(X), axis=-1)

        ans = np.stack(
            [
                MAV,
                RMS,
                VAR,
                STD,
                ZC,
                MPR,
                MAA,
                WL,
                SSC,
                WA,
                MFL,
                KRT,
                SKW,
                MOM3,
                IEMG,
                sampen.squeeze(-1),
                dfa.squeeze(-1),
            ],
            axis=-1,
        )
        return ans.reshape(*ans.shape[:-2], -1)


class DeltaFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Computes raw, delta, and delta^2 along time dimension of EMG windows,
    preserving session structure.

    Input shape:  (n_sessions, n_windows, n_channels, time)
    Output shape: (n_sessions, n_windows, n_channels * 3, time)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, combination=3):
        # X.shape = (n_sessions, n_windows, n_channels, time)
        raw = X  # shape: (S, W, C, T)
        diff = np.diff(X, axis=-1, prepend=X[..., :1])  # Δ
        diff2 = np.diff(diff, axis=-1, prepend=diff[..., :1])  # Δ²

        # Stack along a new "temporal feature" axis: raw, Δ, Δ²
        # Result: (S, W, C, T, 3)
        if combination == 3:
            stacked = np.stack([raw, diff, diff2], axis=-1)
        if combination == 2:
            stacked = np.stack([raw, diff], axis=-1)
        if combination == 1:
            stacked = np.stack([raw, diff2], axis=-1)
        if combination == 0:
            stacked = np.stack([diff, diff2], axis=-1)

        # Rearrange to: (S, W, C * 3, T)
        S, W, C, T, F = stacked.shape
        output = stacked.transpose(0, 1, 2, 4, 3).reshape(S, W, C * F, T)

        return output


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Selects features from the input array.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features) or
                           (n_sessions, n_windows, n_features).

        Returns:
            np.ndarray: Array of selected features, guaranteed to be 2D
                        with shape (n_samples, n_selected_features).
        """
        # Handle 3D shape (n_sessions, n_windows, n_features)
        if X.ndim == 3:
            n_sessions, n_windows, n_features = X.shape
            X = X.reshape(n_sessions * n_windows, n_features)
        elif X.ndim != 2:
            raise ValueError(f"Unsupported input shape: {X.shape}")

        # Defensive check
        if max(self.feature_indices) >= X.shape[1]:
            raise IndexError(
                f"Selected index {max(self.feature_indices)} "
                f"exceeds input feature dimension {X.shape[1]}"
            )

        return X[:, self.feature_indices]


def mutual_info_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return 0.0
    if abs(c) == 1:
        c = 0.999999
    return -0.5 * np.log(1 - c**2)


class TopKMRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k
        self.selected_indices = None

    def fit(self, X, y):
        import numpy as np
        import pandas as pd

        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        if y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        X_df = pd.DataFrame(X)
        y_mat = y

        relevance = []
        for col in X_df.columns:
            rel = np.mean(
                [
                    mutual_info_corr(X_df[col].values, y_mat[:, j])
                    for j in range(y_mat.shape[1])
                ]
            )
            relevance.append(rel)

        selected = []
        candidates = list(range(X_df.shape[1]))

        for _ in range(self.k):
            redundancy = np.zeros(len(candidates))
            if selected:
                for i, c in enumerate(candidates):
                    red = np.mean(
                        [
                            mutual_info_corr(X_df[c].values, X_df[s].values)
                            for s in selected
                        ]
                    )
                    redundancy[i] = red
            scores = np.array(relevance)[candidates] - redundancy
            best = candidates[np.argmax(scores)]
            selected.append(best)
            candidates.remove(best)

        self.selected_indices = selected
        return self

    def transform(self, X):
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        return X[:, self.selected_indices]


class WaveletFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts wavelet-based statistical features from multi-channel time series data.

    Example: For an input shape of (..., 8, n_times) and `level=3` with 4 defined
    metrics, the output shape will be (..., 8 * (1 + 3) * 4), which is (..., 128).

    Attributes:
        wavelet (str): The name of the wavelet function to use (e.g., 'db4', 'haar').
            This is passed to `pywt.wavedec`. Defaults to "db4".
        level (int): The decomposition level for the Discrete Wavelet Transform.
            This determines the number of detail coefficient bands. Defaults to 3.
        metrics (list of callables): A list of functions that accept a 1D numpy
            array (a wavelet coefficient band) and return a scalar feature value.
            The default metrics include standard deviation, Root Mean Square (RMS),
            energy, and spectral entropy.
    """

    def __init__(self, wavelet="db4", level=3):
        self.wavelet = wavelet
        self.level = level
        self.metrics = [
            # lambda w: np.mean(w),
            lambda w: np.std(w),
            lambda w: np.sqrt(np.mean(w**2)),  # RMS
            lambda w: np.sum(w**2),  # Energy
            lambda w: stats.entropy(w**2 / (np.sum(w**2) + 1e-12)),  # Spectral entropy
            # lambda w: stats.kurtosis(w, bias=False),
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X).copy()
        orig_shape = X.shape
        X = X.reshape(
            -1, orig_shape[-2], orig_shape[-1]
        )  # (n_samples, n_channels, n_times)

        feature_list = []
        for window in X:
            ch_feats = []
            for ch in window:
                ch = np.asarray(ch).copy()
                coeffs = pywt.wavedec(ch, self.wavelet, level=self.level)
                band_feats = [np.array([m(b) for m in self.metrics]) for b in coeffs]
                ch_feats.append(np.concatenate(band_feats))
            feature_list.append(np.concatenate(ch_feats))

        return np.array(feature_list).reshape(*orig_shape[:-2], -1)

    def set_output(self, *, transform=None):
        return super().set_output(transform=transform)



class WaveletBandExtractor(BaseEstimator, TransformerMixin):
    """
    Applies Discrete Wavelet Transform (DWT) to each channel of each time window
    and returns the resulting wavelet coefficient bands, padded to a uniform length.

    Input shape:  (n_sessions, n_windows, n_channels, n_times)
    Output shape: (n_sessions, n_windows, n_channels, n_bands, max_band_len)
                  or (n_sessions, n_windows, n_channels * n_bands * max_band_len) if flatten_output=True

    Attributes:
        wavelet (str): The name of the wavelet function to use (e.g., 'db4', 'haar').
            This is passed to `pywt.wavedec`. Defaults to "db4".
        level (int): The decomposition level for the Discrete Wavelet Transform.
            This determines the number of detail coefficient bands. Defaults to 3.
        flatten_output (bool): If True, the output is flattened along the last
            three dimensions (channels, bands, and band length) into a single
            feature vector per window. Defaults to False.
    """

    def __init__(self, wavelet="db4", level=3, flatten_output=False):
        self.wavelet = wavelet
        self.level = level
        self.flatten_output = flatten_output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_sessions, n_windows, n_channels, n_times = X.shape
        X_reshaped = X.reshape(
            -1, n_channels, n_times
        )  # (total_windows, channels, times)

        # Compute max length of any band for padding
        sample_coeffs = pywt.wavedec(np.zeros(n_times), self.wavelet, level=self.level)
        max_len = max(len(b) for b in sample_coeffs)
        n_bands = len(sample_coeffs)

        band_array = np.zeros(
            (X_reshaped.shape[0], n_channels, n_bands, max_len), dtype=np.float32
        )

        for i, window in enumerate(X_reshaped):
            for ch in range(n_channels):
                coeffs = pywt.wavedec(window[ch], self.wavelet, level=self.level)
                for b, band in enumerate(coeffs):
                    band_array[i, ch, b, : len(band)] = band  # pad shorter bands with 0

        # Restore session and window structure
        band_array = band_array.reshape(
            n_sessions, n_windows, n_channels, n_bands, max_len
        )

        if self.flatten_output:
            # Flatten last three dims for (n_sessions, n_windows, features)
            band_array = band_array.reshape(n_sessions, n_windows, -1)

        return band_array

    def set_output(self, *, transform=None):
        return super().set_output(transform=transform)


class SessionwiseTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a base transformer or pipeline independently to each session
    within multi-session data.

    Example Usage:
        riemann_pipeline = Pipeline([
            ('cov', pyriemann.estimation.Covariances()),
            ('ts', pyriemann.tangentspace.TangentSpace(metric='riemann', tsupdate=True)),
        ])

        combined_features = FeatureUnion([
            ("time_features", TimeDomainTransformer()),
            ("riemann_features", riemann_pipeline),
            ("wavelet_features", WaveletFeatureTransformer())
        ])

        sessionwise_combined = SessionwiseTransformer(combined_features)

    Input shape:  (n_sessions, n_windows, n_channels, size)
    Output shape: (n_sessions, n_windows, n_features)
                   where n_features depends on the output of the base_pipeline
                   applied to each session's data (shape: (n_windows, n_features)).

    Attributes:
        base_pipeline (estimator): An instance of a scikit-learn Transformer
            or Pipeline that will be applied to the data of each session.
            This estimator must have `fit_transform` method.
    """

    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X.shape: (n_sessions, n_windows, n_channels, size)
        features = []
        for sess in range(X.shape[0]):
            Xt = self.base_pipeline.fit_transform(X[sess])  # (n_windows, n_features)
            features.append(Xt)
        return np.array(features)  # (n_sessions, n_windows, n_features)
