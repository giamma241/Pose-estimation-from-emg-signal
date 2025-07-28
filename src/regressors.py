import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# AVERAGING ENSEMBLE REGRESSOR
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


# STACKING ENSEMBLE REGRESSOR
class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble voting regressor. Takes the weighted mean of the predictions of other estimators.

    Args:
        estimators (list): list of base regressors
        weights (list, optional): list of weights for averaging. Defaults to uniform.

    Returns:
        np.ndarray: averaged predictions from the ensemble
    """

    def __init__(self, estimators, end_estimator):
        self.estimators = estimators
        self.end_estimator = end_estimator

    def fit(self, X, y):
        est_results = []
        for estimator in self.estimators:
            estimator.fit(X, y)
            est_results.append(estimator.predict(X))
        YY = np.stack(est_results)
        YY = np.moveaxis(YY, 0, -1)
        YY = YY.reshape(*YY.shape[:-2], -1)
        self.end_estimator.fit(YY, y)

    def predict(self, X):
        X = np.asarray(X)
        est_results = []
        for estimator in self.estimators:
            est_results.append(estimator.predict(X))
        YY = np.stack(est_results)
        YY = np.moveaxis(YY, 0, -1)
        YY = YY.reshape(*YY.shape[:-2], -1)
        return self.end_estimator.predict(YY)


class NewStackingRegressor(BaseEstimator, RegressorMixin):
    """
    A stacking regressor that builds an ensemble in two stages: first-level base models generate out-of-fold
    predictions which serve as meta‐features, and a second‐level estimator is then trained on these meta‐features
    to produce the final prediction.

    Parameters
    ----------
    estimators : list of estimators
        A list of fitted scikit-learn–style regressors implementing `fit(X, y)` and `predict(X)`.
    end_estimator : estimator
        A scikit-learn–style regressor to be trained on the meta‐features produced by the base estimators.
    scale_meta_features : bool, default=True
        If True, the meta‐feature matrix is standardized (zero mean, unit variance) before fitting the
        `end_estimator` and before making predictions.
    n_internal_folds : int, default=4
        Number of folds to use when creating out‐of‐fold predictions for the meta‐feature matrix.

    Attributes
    ----------
    meta_features_scaler : StandardScaler, optional
        The scaler used to standardize the meta‐features when `scale_meta_features=True`.
    
    Methods
    -------
    fit(X, y)
        Fit the base estimators in cross‐validated fashion to build meta‐features, fit the scaler (if enabled),
        fit the `end_estimator` on the meta‐features, and then refit all base estimators on the full training data.
    predict(X)
        Generate base‐level predictions, form meta‐features (applying scaling if enabled), and return the
        final prediction from the fitted `end_estimator`.

    Returns
    -------
    self : object
        Fitted estimator.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> from NewStackingRegressor import NewStackingRegressor
    >>> base_learners = [LinearRegression(), Ridge(alpha=1.0)]
    >>> meta_model = LinearRegression()
    >>> stack = NewStackingRegressor(estimators=base_learners,
    ...                             end_estimator=meta_model,
    ...                             scale_meta_features=True,
    ...                             n_internal_folds=5)
    >>> stack.fit(X_train, y_train)
    NewStackingRegressor(...)
    >>> y_pred = stack.predict(X_test)
    """
    def __init__(
            self,
            estimators,
            end_estimator,
            scale_meta_features = True,
            n_internal_folds = 4):
        self.estimators = estimators
        self.end_estimator = end_estimator
        self.scale_meta_features = scale_meta_features
        if scale_meta_features:
            self.meta_features_scaler = StandardScaler()
        self.n_internal_folds = n_internal_folds

    def fit(self, X, Y):
        
        n_folds = self.n_internal_folds # getting number of folds for internal cv
        length_fold = X.shape[0] // n_folds # getting length of each fold
        X_folds = X.reshape(n_folds, length_fold, *X.shape[1:]) # splitting X in folds
        Y_folds = Y.reshape(n_folds, length_fold, *Y.shape[1:]) # splitting Y in folds
        
        meta_features_folds = [] # initializing matrix of meta-features
        meta_labels_folds = [] # initializing meta-labels
        for fold in range(n_folds):
            idx_train_folds = [i for i in range(n_folds)]
            idx_train_folds.remove(fold)

            # flattening the training folds of X and Y
            X_train = X_folds[idx_train_folds].reshape(-1, *X_folds.shape[2:])
            Y_train = Y_folds[idx_train_folds].reshape(-1, *Y_folds.shape[2:])

            # separating validation folds
            X_val = X_folds[fold]
            meta_labels_folds.append(Y_folds[fold])
            
            # fitting base estimators and populating the matrix of meta-features
            Y_val_est = []
            for base_est in self.estimators:
                base_est.fit(X_train, Y_train)
                Y_val_est.append(base_est.predict(X_val))
            meta_features_folds.append(np.hstack(Y_val_est))

        # stacking meta-features and meta-labels
        meta_features = np.vstack(meta_features_folds)
        meta_labels = np.vstack(meta_labels_folds)

        # scaling meta-features
        if self.scale_meta_features:
            meta_features = self.meta_features_scaler.fit_transform(meta_features)

        # fitting the meta estimator
        self.end_estimator.fit(meta_features, meta_labels)

        # re-fitting base learners on the whole dataset
        for base_est in self.estimators:
            base_est.fit(X, Y)

        return self
            

    def predict(self, X):
        # generating meta-features
        Y_pred_est = []
        for base_est in self.estimators:
            Y_pred_est.append(base_est.predict(X))
        meta_features = np.hstack(Y_pred_est)

        # generating the meta-prediction
        if self.scale_meta_features:
            meta_features = self.meta_features_scaler.transform(meta_features)
        Y_pred = self.end_estimator.predict(meta_features)

        return Y_pred

# NEURAL NETWORK REGRESSOR
class NNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            model_class,
            model_parameters,
            loss_fnc,
            batch_size,
            learning_rate,
            max_epochs,
            patience):
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.loss_fnc = loss_fnc
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_model = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def fit(self, X, y):
        # initialize model
        model = self.model_class(**self.model_parameters)
        model.build()
        model = model.to(self.device)
        
        # create datasets and dataloaders
        XX = torch.tensor(X, dtype = torch.float, device=self.device)
        yy = torch.tensor(y, dtype = torch.float, device=self.device)
        train_dataset = TensorDataset(XX, yy)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # initialize optimizer
        criterion = self.loss_fnc
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # training loop
        best_train_loss = np.inf
        for epoch in range(1, self.max_epochs + 1):
            
            model.train()
            total_train_loss = 0.0
            for Xb, Yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, Yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                epochs_no_improve = 0
                self.best_model = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break
        return self

    def predict(self, X):
        # initialize model
        model = self.model_class(**self.model_parameters)
        model.build()
        model = model.to(self.device)
        
        # load best model parameters
        model.load_state_dict(self.best_model)

        # compute predictions
        XX = torch.tensor(X, dtype=torch.float, device=self.device)
        YY = model(XX)
        Y = YY.detach().cpu().numpy()
        return Y

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        # initialize model
        model = self.model_class(**self.model_parameters)
        model.build()
        model = model.to(self.device)
        
        # create datasets and dataloaders
        XX_train = torch.tensor(X_train, dtype = torch.float, device=self.device)
        yy_train = torch.tensor(y_train, dtype = torch.float, device=self.device)
        XX_val = torch.tensor(X_val, dtype = torch.float, device=self.device)
        yy_val = torch.tensor(y_val, dtype = torch.float, device=self.device)
        train_dataset = TensorDataset(XX_train, yy_train)
        val_dataset = TensorDataset(XX_val, yy_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # initialize optimizer
        criterion = self.loss_fnc
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # training loop
        train_losses = []
        val_losses = []
        best_train_loss = np.inf
        for epoch in range(1, self.max_epochs + 1):
            
            model.train()
            total_train_loss = 0.0
            for Xb, Yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, Yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for Xb, Yb in val_loader:
                    preds = model(Xb)
                    loss = criterion(preds, Yb)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | ")

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                epochs_no_improve = 0
                self.best_model = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break
            
        return (np.asarray(train_losses), np.asarray(val_losses))