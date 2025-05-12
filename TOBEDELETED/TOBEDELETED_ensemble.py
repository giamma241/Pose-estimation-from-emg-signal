import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class VotingRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble voting regressor. Takes the weighted mean of the predictions of other estimators.

    Args:
        estimators (list): list of base regressors
        weights (list, optional): list of weights for averaging. Defaults to uniform.

    Returns:
        np.ndarray: averaged predictions from the ensemble
    """

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        if weights is None:
            self.weights = [1 / len(estimators)] * len(estimators)
        else:
            self.weights = weights

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = np.asarray(
            [estimator.predict(X) for estimator in self.estimators]
        )
        return np.average(predictions, weights=self.weights, axis=0)