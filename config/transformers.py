import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import butter, decimate, filtfilt, iirnotch, resample, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from config.validation import mutual_info_corr


class EmgFilterTransformer:
    def __init__(self, original_fs = 1024, target_fs = 2048, f0 = 50.0, bw = 5.0, low = 30.0, high = 500.0, order = 4):
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.f0 = f0 
        self.bw = bw
        self.low = low
        self.high = high
        self.order = order

    def transform(self, X):
        """
        Resample signal using anti-aliasing filters
        Apply notch and zero-phase bandpass filters to EMG signal.
        Parameters:
            emg_signal (array): Raw EMG signal
            fs (float): Sampling frequency in Hz
            
        Returns:
            array: Resampled signal
        
        """
        # resample
        X_resampled = self._resample_signal(X, self.original_fs, self.target_fs)
        
        # 1. Notch filter
        Q = self.f0 / self.bw    # Quality factor
        b_notch, a_notch = iirnotch(self.f0, Q, self.target_fs)
        X_notched = filtfilt(b_notch, a_notch, X_resampled)
        
        # 2. Bandpass filter (30-500 Hz)
        sos_bandpass = butter(
            self.order,
            [self.low, self.high],
            btype='bandpass',
            fs=self.target_fs,
            output='sos')
        X_filtered = sosfiltfilt(sos_bandpass, X_notched)

        # resample back
        X_filtered_resampled = self._resample_signal(X_filtered, self.target_fs, self.original_fs)
        
        return X_filtered_resampled
    
    def _resample_signal(self, signal, original_fs, target_fs):
        """
        Resample signal using anti-aliasing filters
        
        Parameters:
            signal (array): Input signal
            original_fs (float): Original sampling frequency
            target_fs (float): Target sampling frequency
            
        Returns:
            array: Resampled signal
        """
        ratio = original_fs / target_fs
        
        if ratio > 1:  # Downsampling
            # Ensure integer ratio
            if not ratio.is_integer():
                raise ValueError("Use Method 2 for non-integer ratios")
            resampled = decimate(signal, int(ratio), zero_phase=True)
            
        elif ratio < 1:  # Upsampling
            # Use Fourier method for best time-domain preservation
            num_samples = int(len(signal) * target_fs / original_fs)
            resampled = resample(signal, num_samples)
            
        return resampled


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
    Transformer that extracts N standard time-domain features
    from the last axis of a NumPy array of shape (..., n_times),
    returning an array of shape (..., N).
    """
    def __init__(self, sigma_mpr=1.0):
        """
        Parameters:
            sigma_mpr (float, optional):    value of sigma for the Myopulse Percentage Rate.
                                            Defaults to 1.0.
        """
        self.sigma_mpr = sigma_mpr

    def fit(self, X, y=None):
        # does nothing, here for compatibility
        return self

    def transform(self, X):
        """
        Args:
            X : np.ndarray, shape (...,n_times)

        Returns:
            np.ndarray: the array of EMG time domain features of shape (...,n_features)
                         computed on the time dimension of X
                         feature list:
                         [MAV, RMS, VAR, STD, ZC, MPR, MAA, WL, SSC, WA, MFL, KRT]      
        """
        MAV = np.mean(np.abs(X), axis=-1) # Mean Absolute Value
        RMS = np.sqrt(np.mean(X**2, axis=-1)) # Root Mean Square
        VAR = np.var(X, ddof=1, axis=-1) # Variance
        STD = np.std(X, ddof=1, axis=-1) # Standard Deviation
        ZC  = np.sum(X[..., :-1] * X[..., 1:] < 0, axis=-1) # Zero crossing count
        MPR = np.mean(np.abs(X) > self.sigma_mpr, axis=-1) # Myopulse percentage rate
        MAA = np.max(np.abs(X), axis=-1) # Maximum absolute amplitude
        WL  = np.sum(np.abs(np.diff(X, axis=-1)), axis=-1) # Waveform length
        
        SSC = np.sum(
            (-(X[..., 2:] - X[..., 1:-1]) *
              (X[..., 1:-1] - X[..., :-2]) > 0),
            axis=-1
        ) # slope sign changes
        WA  = np.sum(
            np.abs(np.diff(X, axis=-1)) - STD[..., None] > 0,
            axis=-1
        ) # Wilson amplitude
        MFL = np.log(
            np.sqrt(np.sum(np.diff(X, axis=-1)**2, axis=-1))
        ) # Maximum fractal length
        KRT = stats.kurtosis(X, axis=-1, bias=False) # Kurtosis
        
        ans = np.stack([MAV, RMS, VAR, STD, ZC, MPR,
                         MAA, WL, SSC, WA, MFL, KRT],
                        axis=-1)
        return ans.reshape(*ans.shape[:-2], -1)
    
    def set_output(self, *, transform=None):
        # Here for compatibility
        return super().set_output(transform=transform)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_windows = X.shape[1]
        return X.reshape(X.shape[0] * n_windows, -1)[:, self.feature_indices]


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