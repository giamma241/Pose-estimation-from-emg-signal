###########
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("../")

#### LIBRARIES


from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Function
from torch.utils.data import Dataset

from config.regressors import *
from config.transformers import *
from config.validation import NMSE, RMSE


class NMSELoss(nn.Module):
    def _init_(self, eps: float = 1e-8):
        super()._init_()

    def forward(self, y_pred, y_true):
        num = (y_true - y_pred).pow(2).sum(dim=1).mean()
        den = (y_true - y_true.mean(dim=0)).pow(2).sum(dim=1).mean()
        return num / den


class RMSELoss(nn.Module):
    def _init_(self, eps=1e-8):
        super()._init_()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        return torch.sqrt(mse_loss)


##### Convolutional Neural Network
class ConvNN(nn.Module):
    def _init_(self, end_dim):
        super()._init_()

        # Feature extractor: temporal + spatial + temporal blocks
        self.features = nn.Sequential(
            # 1. Temporal Convolution
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 2. Spatial Convolution Block
            nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(p=0.3),
            # 3. Temporal Convolution Block
            nn.Conv1d(in_channels=16, out_channels=128, kernel_size=16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(p=0.3),
        )

        # Global pooling to reduce temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Regression head for 3D output
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=end_dim),
        )

    def forward(self, x):
        # x: (batch_size, 8, seq_len)
        x = self.features(x)
        # x: (batch_size, 128, seq_len_reduced)
        x = self.global_pool(x).squeeze(-1)
        # x: (batch_size, 128)
        out = self.fc(x)
        return out

    def build(self):
        pass  # here for compatibility


# ======= Dataset =======
class DANNWindowTensor(Dataset):
    def _init_(self, X, Y, session_ids):
        # Detect and reshape if unflattened
        if X.ndim == 4:  # (sessions, windows, channels, time)
            X = X.reshape(-1, *X.shape[2:])  # (N, 8, 500)
        if Y.ndim == 3:  # (sessions, windows, features)
            Y = Y.reshape(-1, Y.shape[-1])  # (N, 51)

        if len(session_ids) != len(X):
            raise ValueError(
                f"session_ids ({len(session_ids)}) must match number of samples ({len(X)})"
            )

        self.X = X
        self.Y = Y
        self.session_ids = torch.tensor(session_ids, dtype=torch.long)

    def _len_(self):
        return len(self.X)

    def _getitem_(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            self.session_ids[idx],
        )


# ======= Model Components =======


class ConvFeatureExtractor(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Conv1d(8, 128, 16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
            nn.Dropout(0.3),
            nn.Conv1d(16, 128, 16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
            nn.Dropout(0.3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.net(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class TemporalConvFeatureExtractor(nn.Module):
    def _init_(self, in_channels=24):
        super()._init_()
        self.net = nn.Sequential(
            # First block
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),  # (B, 64, 500)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Second block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # (B, 128, 500)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool1d(kernel_size=2),  # (B, 128, 250)
            # Third block
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AvgPool1d(kernel_size=2),  # (B, 128, 125)
        )

        # === Global pooling for sequence summarisation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 128, 1)

    def forward(self, x):
        x = self.net(x)  # shape: (B, 128, T')
        x = self.global_pool(x).squeeze(-1)  # shape: (B, 128)
        return x


class TemporalConvFeatureExtractor2(nn.Module):
    def _init_(self, in_channels=24, groups=8):
        """
        Feature extractor with:
        - grouped convs (Hu et al., 2018; Wang et al., 2020)
        - long kernel in first layer (Hannink et al., 2017)
        - dilation for receptive field expansion (Bai et al., 2018)
        - dropout for generalisation
        """
        super()._init_()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=31,
                padding=15,
                groups=groups,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # === Global pooling for sequence summarisation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Input:  (B, C, T)
        Output: (B, 128)
        """
        x = self.net(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class RegressorHead(nn.Module):
    def _init_(self, input_dim=128, output_dim=51):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DomainDiscriminator(nn.Module):
    def _init_(self, input_dim=128, num_domains=5):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_domains),
        )

    def forward(self, x):
        return self.net(x)


# ======= Gradient Reversal =======
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversalLayer.apply(x, lambda_)


# ======= DANN Models =======
class DANNModel(nn.Module):
    def _init_(self, lambda_grl=1.0, num_domains=5, output_dim=51):
        super()._init_()
        self.feature_extractor = ConvFeatureExtractor()
        self.regressor_head = RegressorHead(128, output_dim)
        self.domain_discriminator = DomainDiscriminator(128, num_domains)
        self.grl = GradientReversalLayer(lambda_=lambda_grl)

    def forward(self, x, lambda_grl=1.0):
        features = self.feature_extractor(x)
        y_pred = self.regressor_head(features)
        domain_pred = self.domain_discriminator(grad_reverse(features, lambda_grl))
        return y_pred, domain_pred


class TemporalDANNModel(nn.Module):
    def _init_(
        self, lambda_grl=1.0, num_domains=5, output_dim=51, in_channels=24, mode=1
    ):
        super()._init_()
        if mode == 1:
            self.feature_extractor = TemporalConvFeatureExtractor(
                in_channels=in_channels
            )
        if mode == 2:
            self.feature_extractor = TemporalConvFeatureExtractor2(
                in_channels=in_channels
            )
        self.regressor_head = RegressorHead(input_dim=128, output_dim=output_dim)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=128, num_domains=num_domains
        )

    def forward(self, x, lambda_grl=1.0):
        features = self.feature_extractor(x)
        y_pred = self.regressor_head(features)
        domain_pred = self.domain_discriminator(grad_reverse(features, lambda_grl))
        return y_pred, domain_pred


# ======= Model Regressor Class =======


class DANNRegressor(BaseEstimator, RegressorMixin):
    def _init_(
        self,
        model,
        session_ids=None,
        batch_size=128,
        max_epochs=50,
        patience=10,
        learning_rate=1e-3,
        lambda_grl=1.0,
        gamma_entropy=0.0,
        device="cuda",
        verbose=True,
        dataset_class=None,
    ):
        self.model = model
        self.session_ids = session_ids
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.lambda_grl = lambda_grl
        self.gamma_entropy = gamma_entropy
        self.device = device
        self.verbose = verbose
        self.dataset_class = dataset_class
        self.trainer = None

    def fit(self, X, Y):
        if self.session_ids is None:
            raise ValueError("session_ids must be set before calling fit().")
        dataset = self.dataset_class(X, Y, self.session_ids)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.trainer = DANNTrainer(
            model=self.model,
            X=None,
            Y=None,
            train_sessions=[],
            val_session=None,  # unused
            lambda_grl=self.lambda_grl,
            gamma_entropy=self.gamma_entropy,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            learning_rate=self.learning_rate,
            device=self.device,
            tensor_dataset=self.dataset_class,
        )
        self.trainer.train_loader = train_loader
        self.trainer.train(validate=False)
        return self

    def predict(self, X_test_windows):
        X_predictors = DataLoader(
            TensorDataset(torch.tensor(X_test_windows, dtype=torch.float32)),
            batch_size=128,
            shuffle=False,
        )
        return self.trainer.predict(X_predictors)


class DANNTrainer:
    def _init_(
        self,
        model: nn.Module,
        X: np.ndarray,
        Y: np.ndarray,
        train_sessions: list,
        val_session: int,
        lambda_grl: float = 0.1,
        gamma_entropy=0.1,
        batch_size: int = 128,
        max_epochs: int = 50,
        patience: int = 10,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tensor_dataset=None,
        train_loader=None,
        val_loader=None,
        verbose=True,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.lambda_grl = lambda_grl
        self.gamma_entropy = gamma_entropy
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reg_loss = nn.MSELoss()
        self.dom_loss = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.train_losses = []  # Added
        self.val_losses = []  # Added

    def entropy_loss(self, logits):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        return -torch.sum(probs * log_probs, dim=1).mean()

    def train(self, validate=True):
        verbose = self.verbose
        best_val_rmse = float("inf")
        best_weights = None
        patience_counter = 0
        self.train_losses = []  # Reset each training run
        self.val_losses = []
        self.train_nmses = []

        def compute_lambda(epoch, max_lambda=1.0, warmup_epochs=10):
            return min(max_lambda, epoch / warmup_epochs)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_y_true, train_y_pred = [], []
            epoch_losses = []

            lambda_grl = compute_lambda(epoch)

            for x, y, sid in self.train_loader:
                x, y, sid = x.to(self.device), y.to(self.device), sid.to(self.device)
                y_pred, dom_pred = self.model(x, lambda_grl=lambda_grl)

                loss = self.reg_loss(y_pred, y) + lambda_grl * self.dom_loss(
                    dom_pred, sid
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

                train_y_true.append(y.detach().cpu().numpy())
                train_y_pred.append(y_pred.detach().cpu().numpy())

            train_y_true = np.vstack(train_y_true)
            train_y_pred = np.vstack(train_y_pred)
            train_rmse = RMSE(train_y_pred, train_y_true)
            train_nmse = NMSE(train_y_pred, train_y_true)

            self.train_losses.append(train_rmse)
            self.train_nmses.append(train_nmse)

            if validate:
                val_rmse = self.evaluate()
                self.val_losses.append(val_rmse)
                domain_train_acc = self.evaluate_domain_on_train()

                if verbose:
                    print(
                        f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Train NMSE: {train_nmse:.4f} "
                        f"| Val RMSE: {val_rmse:.4f} | Dom Train Acc: {domain_train_acc:.2%}"
                    )
                    print(f"           λ = {lambda_grl:.3f}")

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_weights = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        break
            else:
                if verbose:
                    print(
                        f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Train NMSE: {train_nmse:.4f}"
                    )
                    print(f"           λ = {lambda_grl:.3f}")

        if validate and best_weights is not None:
            self.model.load_state_dict(best_weights)
        return (
            {
                "epoch": list(range(1, len(self.train_losses) + 1)),
                "Train RMSE": self.train_losses,
                "Val RMSE": self.val_losses,
            }
            if validate
            else self.model
        )

    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x, *_ in data_loader:
                x = x.to(self.device)
                preds = self.model(x, lambda_grl=self.lambda_grl)[0]
                all_preds.append(preds.cpu().numpy())
        return np.vstack(all_preds)

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                yp, _ = self.model(x)
                y_pred.append(yp.cpu().numpy())
                y_true.append(y.cpu().numpy())
        return RMSE(np.vstack(y_pred), np.vstack(y_true))

    def evaluate_domain_on_train(self):
        self.model.eval()
        s_true, s_pred = [], []
        with torch.no_grad():
            for x, _, sid in self.train_loader:
                x, sid = x.to(self.device), sid.to(self.device)
                _, dom_out = self.model(x)
                pred_sid = torch.argmax(dom_out, dim=1)
                s_pred.append(pred_sid.cpu().numpy())
                s_true.append(sid.cpu().numpy())
        return (np.concatenate(s_pred) == np.concatenate(s_true)).mean()