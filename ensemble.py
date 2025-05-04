import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score


class BasicTransformer(BaseEstimator, TransformerMixin):
    """
    Identity transformer for testing or bypassing preprocessing steps.
    Returns input unchanged.

    Returns:
        np.ndarray: unmodified input array
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def set_output(self, *, transform=None):
        return super().set_output(transform=transform)


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

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        Args:
            X (np.ndarray): input features
            y (np.ndarray): ground truth labels
            sample_weight (np.ndarray, optional): sample weights

        Returns:
            float: R^2 score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def set_output(self, *, predict=None):
        return super().set_output(predict=predict)
