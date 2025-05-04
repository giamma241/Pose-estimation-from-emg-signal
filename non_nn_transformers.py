import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics import r2_score

class BasicTransformer(BaseEstimator, TransformerMixin):
    """
    Template.
    """
    def __init__(self):
        # No hyperparameters yet
        pass

    def fit(self, X, y=None):
        # Nothing to learn
        return self

    def transform(self, X):
        # Return input as-is
        return X

    def set_output(self, *, transform=None):
        # Here for compatibility
        return super().set_output(transform=transform)


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


class VotingRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble voting regressor. Takes the weighted mean of the predictions of other estimators.
    """
    def __init__(self, estimators, weights = None):
        """
        estimators and weights are lists
        """
        self.estimators = estimators
        if weights == None:
            self.weights = [1/len(estimators) for  _ in range(len(estimators))]
        else:
            self.weights = weights

    def fit(self, X, y):
        """
        """
        for estimator in self.estimators:
            estimator.fit(X,y)
        return self

    def predict(self, X):
        """
        """
        X = np.asarray(X)
        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators])
        return np.average(predictions, weights = self.weights, axis = 0)

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def set_output(self, *, predict=None):
        """
        Set output container object type for `predict`.
        """
        return super().set_output(predict=predict)
