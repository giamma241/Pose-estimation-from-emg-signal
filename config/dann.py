import sys

sys.path.append("../")

#### LIBRARIES


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from config.loss_functions import *
from config.models import *
from config.regressors import *
from config.transformers import *
from config.validation import NMSE, RMSE
from matplotlib.lines import Line2D
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset


# ======= Dataset =======
class DANNWindowTensor(Dataset):
    def __init__(self, X, Y, session_ids):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            self.session_ids[idx],
        )


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


# ======= Model Components =======


class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
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
    def __init__(self, in_channels=24):
        super().__init__()
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

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 128, 1)

    def forward(self, x):
        x = self.net(x)  # shape: (B, 128, T')
        x = self.global_pool(x).squeeze(-1)  # shape: (B, 128)
        return x


import torch.nn as nn


class TemporalConvFeatureExtractor2(nn.Module):
    def __init__(self, in_channels=24, groups=8):
        """
        Feature extractor with:
        - grouped convs (Hu et al., 2018; Wang et al., 2020)
        - long kernel in first layer (Hannink et al., 2017)
        - dilation for receptive field expansion (Bai et al., 2018)
        - dropout for generalisation
        """
        super().__init__()

        self.net = nn.Sequential(
            # === Grouped convolution: one group per sensor
            # Each group sees [raw, Δ, Δ²] for a single electrode
            # Cited: Hu et al., 2018 (DeepEMG); Wang et al., 2020 (MuscleNet)
            nn.Conv1d(
                in_channels=in_channels,  # typically 24 = 8 sensors × 3
                out_channels=64,
                kernel_size=31,  # large receptive field (~60ms at 500Hz)
                padding=15,
                groups=groups,  # 8 = number of physical sensors
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # === Dilated conv to expand receptive field without pooling
            # Cited: Bai et al., 2018 (dilated conv > RNNs for sequence modeling)
            nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # === Strided convolution for downsampling (replaces pooling)
            # Cited: Hannink et al., 2017 (gait analysis), preserves phase info
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
    def __init__(self, input_dim=128, output_dim=51):
        super().__init__()
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
    def __init__(self, input_dim=128, num_domains=5):
        super().__init__()
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


# ======= Full DANN Model =======
class DANNModel(nn.Module):
    def __init__(self, lambda_grl=1.0, num_domains=5, output_dim=51):
        super().__init__()
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
    def __init__(
        self, lambda_grl=1.0, num_domains=5, output_dim=51, in_channels=24, mode=1
    ):
        super().__init__()
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


# ===== REGRESSOR CLASS ======


# class DANNRegressor(BaseEstimator, RegressorMixin):
#     def __init__(
#         self,
#         model,
#         session_ids=None,
#         batch_size=128,
#         max_epochs=50,
#         patience=10,
#         learning_rate=1e-3,
#         lambda_grl=1.0,
#         gamma_entropy=0.0,
#         device="cuda",
#         verbose=True,
#         dataset_class=None,
#     ):
#         self.model = model
#         self.session_ids = session_ids
#         self.batch_size = batch_size
#         self.max_epochs = max_epochs
#         self.patience = patience
#         self.learning_rate = learning_rate
#         self.lambda_grl = lambda_grl
#         self.gamma_entropy = gamma_entropy
#         self.device = device
#         self.verbose = verbose
#         self.dataset_class = dataset_class
#         self.trainer = None

#     def fit(self, X, Y):
#         if self.session_ids is None:
#             raise ValueError("session_ids must be set before calling fit().")

#         dataset = self.dataset_class(X, Y, self.session_ids)
#         train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         config = DANNConfig(
#             batch_size=self.batch_size,
#             max_epochs=self.max_epochs,
#             patience=self.patience,
#             learning_rate=self.learning_rate,
#             lambda_grl=self.lambda_grl,
#             gamma_entropy=self.gamma_entropy,
#             device=self.device,
#             verbose=2 if self.verbose else 0,  # Map to our verbosity levels
#             num_domains=len(np.unique(self.session_ids)),  # critical for DANN
#             output_dim=Y.shape[-1],
#         )

#         self.trainer = DANNTrainer(
#             model=self.model, train_loader=train_loader, val_loader=None, config=config
#         )
#         self.trainer.train(validate=False)
#         return self

#     def predict(self, X_test_windows):
#         loader = DataLoader(
#             TensorDataset(torch.tensor(X_test_windows, dtype=torch.float32)),
#             batch_size=self.batch_size,
#             shuffle=False,
#         )
#         return self.trainer.predict(loader)


# import torch
# import torch.nn as nn


# @dataclass
# class DANNConfig:
#     batch_size: int = 128
#     max_epochs: int = 50
#     patience: int = 10
#     learning_rate: float = 1e-3
#     lambda_grl: float = 1.0
#     gamma_entropy: float = 0.0
#     output_dim: int = 51
#     device: str = "cuda"
#     verbose: int = 1  # 0: silent, 1: fold summary + plots, 2: epoch logs


# class DANNTrainer:
#     def __init__(
#         self, model: nn.Module, train_loader, val_loader=None, config: DANNConfig = None
#     ):
#         self.model = model.to(config.device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.config = config

#         self.lambda_grl = config.lambda_grl
#         self.gamma_entropy = config.gamma_entropy
#         self.max_epochs = config.max_epochs
#         self.patience = config.patience
#         self.batch_size = config.batch_size
#         self.learning_rate = config.learning_rate
#         self.device = config.device
#         self.verbose = config.verbose

#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=config.learning_rate
#         )
#         self.reg_loss = nn.MSELoss()
#         self.dom_loss = nn.CrossEntropyLoss()

#         self.train_losses = []
#         self.val_losses = []
#         self.train_nmses = []

#     def _compute_grl_lambda(self, epoch):
#         """
#         Sigmoid ramp-up schedule, capped at config.lambda_grl
#         """
#         p = epoch / self.max_epochs
#         lam = 2.0 / (1 + np.exp(-10 * p)) - 1
#         return min(lam, self.config.lambda_grl)

#     def train(self, validate=True):
#         best_val_rmse = float("inf")
#         best_weights = None
#         patience_counter = 0

#         for epoch in range(1, self.max_epochs + 1):
#             lambda_grl = self._compute_grl_lambda(epoch)
#             self.lambda_grl = lambda_grl
#             train_rmse, train_nmse = self._train_epoch(lambda_grl)
#             self.train_losses.append(train_rmse)
#             self.train_nmses.append(train_nmse)

#             if validate:
#                 val_rmse, dom_acc = self._validate_epoch(lambda_grl)
#                 self.val_losses.append(val_rmse)

#                 if self.verbose:
#                     print(
#                         f"Epoch {epoch:03d} | Train Loss: {train_rmse:.4f} | Val Loss: {val_rmse:.4f} | Dom Train Acc: {dom_acc:.2%}"
#                     )

#                 if val_rmse < best_val_rmse:
#                     best_val_rmse = val_rmse
#                     best_weights = self.model.state_dict()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= self.patience:
#                         print("Early stopping triggered.")
#                         break
#             elif self.verbose:
#                 print(f"Epoch {epoch:03d} | Train Loss: {train_rmse:.4f}")

#         if validate and best_weights is not None:
#             self.model.load_state_dict(best_weights)

#         return best_val_rmse if validate else self.model

#     def _train_epoch(self, lambda_grl):
#         self.model.train()
#         y_true, y_pred = [], []

#         for x, y, sid in self.train_loader:
#             x, y, sid = x.to(self.device), y.to(self.device), sid.to(self.device)
#             yp, dom_pred = self.model(x, lambda_grl=lambda_grl)

#             loss = self.reg_loss(yp, y) + lambda_grl * self.dom_loss(dom_pred, sid)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             y_true.append(y.detach().cpu().numpy())
#             y_pred.append(yp.detach().cpu().numpy())

#         y_true = np.vstack(y_true)
#         y_pred = np.vstack(y_pred)
#         return RMSE(y_pred, y_true), NMSE(y_pred, y_true)

#     def _validate_epoch(self, lambda_grl):
#         self.model.eval()
#         y_true, y_pred = [], []
#         s_true, s_pred = [], []

#         with torch.no_grad():
#             for x, y, sid in self.val_loader:
#                 x, y, sid = x.to(self.device), y.to(self.device), sid.to(self.device)
#                 yp, dom_out = self.model(x, lambda_grl=lambda_grl)
#                 y_pred.append(yp.cpu().numpy())
#                 y_true.append(y.cpu().numpy())

#                 pred_sid = torch.argmax(dom_out, dim=1)
#                 s_pred.append(pred_sid.cpu().numpy())
#                 s_true.append(sid.cpu().numpy())

#         y_true = np.vstack(y_true)
#         y_pred = np.vstack(y_pred)
#         dom_acc = (np.concatenate(s_pred) == np.concatenate(s_true)).mean()

#         return RMSE(y_pred, y_true), dom_acc

#     def predict(self, data_loader):
#         self.model.eval()
#         preds = []
#         with torch.no_grad():
#             for x, *_ in data_loader:
#                 x = x.to(self.device)
#                 yp, _ = self.model(x, lambda_grl=self.lambda_grl)
#                 preds.append(yp.cpu().numpy())
#         return np.vstack(preds)

#     def evaluate(self):
#         return self._validate_epoch(lambda_grl=self.lambda_grl)[0]  # only return RMSE

#     def evaluate_domain_on_train(self):
#         self.model.eval()
#         s_true, s_pred = [], []
#         with torch.no_grad():
#             for x, _, sid in self.train_loader:
#                 x, sid = x.to(self.device), sid.to(self.device)
#                 _, dom_out = self.model(x)
#                 pred_sid = torch.argmax(dom_out, dim=1)
#                 s_pred.append(pred_sid.cpu().numpy())
#                 s_true.append(sid.cpu().numpy())
#         return (np.concatenate(s_pred) == np.concatenate(s_true)).mean()


# @dataclass
# class DANNConfig:
#     batch_size: int = 128
#     max_epochs: int = 50
#     patience: int = 10
#     learning_rate: float = 1e-3
#     lambda_grl: float = 1.0
#     gamma_entropy: float = 0.0
#     num_domains: int = 5
#     output_dim: int = 51
#     device: str = "cuda"
#     verbose: int = 1  # 0: silent, 1: fold summary + plots, 2: epoch logs


# class DANNRunner:
#     def __init__(self, model_class, dataset_class, config: DANNConfig):
#         self.model_class = model_class
#         self.dataset_class = dataset_class
#         self.config = config
#         self.fold_stats = {}

#     def _prepare_data(self, X, Y, train_idx, val_idx):
#         session_ids_train = np.concatenate(
#             [np.full(X[s].shape[0], i) for i, s in enumerate(train_idx)]
#         )
#         train_dataset = self.dataset_class(
#             X[train_idx], Y[train_idx], session_ids_train
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=self.config.batch_size, shuffle=True
#         )

#         X_val = X[val_idx]
#         Y_val = Y[val_idx]
#         session_ids_val = np.full(X_val.shape[0], val_idx)
#         val_dataset = self.dataset_class(X_val, Y_val, session_ids_val)
#         val_loader = DataLoader(
#             val_dataset, batch_size=self.config.batch_size, shuffle=False
#         )

#         return train_loader, val_loader

#     def _plot_losses(self, train_losses, val_losses, fold):
#         fig = plt.figure()
#         plt.plot(train_losses, label="Training", color="blue", marker="s")
#         plt.plot(val_losses, label="Validation", color="red", marker="o")
#         plt.title(f"Average batch losses per epoch - Fold {fold + 1}")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.grid(True)
#         plt.legend(
#             handles=[
#                 Line2D([0], [0], color="red", marker="o", linestyle="-"),
#                 Line2D([0], [0], color="blue", marker="s", linestyle="-"),
#             ],
#             labels=["Validation", "Training"],
#             title="Groups",
#         )
#         plt.show()

#     def _evaluate_fold(self, trainer, val_loader):
#         train_rmse = trainer.train_losses[-1]
#         train_nmse = trainer.train_nmses[-1]

#         val_preds = trainer.predict(val_loader)
#         val_true = np.vstack([y for _, y, _ in val_loader.dataset])
#         val_rmse = RMSE(val_preds, val_true)
#         val_nmse = NMSE(val_preds, val_true)

#         return train_rmse, train_nmse, val_rmse, val_nmse

#     def cross_validate(self, X, Y):
#         train_rmses, val_rmses = [], []
#         train_nmses, val_nmses = [], []

#         for val_session in range(X.shape[0]):
#             train_sessions = [s for s in range(X.shape[0]) if s != val_session]

#             if self.config.verbose >= 1:
#                 print(
#                     f"\n=== Fold {val_session + 1} | Train on {train_sessions}, Validate on {val_session} ==="
#                 )

#             train_loader, val_loader = self._prepare_data(
#                 X, Y, train_sessions, val_session
#             )
#             model = self.model_class(
#                 num_domains=len(train_sessions),
#                 output_dim=self.config.output_dim,
#             )

#             trainer = DANNTrainer(
#                 model=model,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 config=self.config,
#             )

#             trainer.train()
#             train_rmse, train_nmse, val_rmse, val_nmse = self._evaluate_fold(
#                 trainer, val_loader
#             )

#             self.fold_stats[f"Fold {val_session + 1}"] = {
#                 "RMSE": {"train": train_rmse, "val": val_rmse},
#                 "NMSE": {"train": train_nmse, "val": val_nmse},
#             }

#             train_rmses.append(train_rmse)
#             val_rmses.append(val_rmse)
#             train_nmses.append(train_nmse)
#             val_nmses.append(val_nmse)

#             if self.config.verbose == 1:
#                 self._plot_losses(trainer.train_losses, trainer.val_losses, val_session)
#                 print(f"\nFold {val_session + 1}")
#                 print(f"RMSE: train={train_rmse:.4f}, val={val_rmse:.4f}")
#                 print(f"NMSE: train={train_nmse:.4f}, val={val_nmse:.4f}")

#         self.fold_stats["Average"] = {
#             "RMSE": {"train": np.mean(train_rmses), "val": np.mean(val_rmses)},
#             "NMSE": {"train": np.mean(train_nmses), "val": np.mean(val_nmses)},
#         }

#         if self.config.verbose in [1, 2]:
#             print("\nAverage Scores across folds:")
#             print(
#                 f"RMSE: train={np.mean(train_rmses):.4f}, val={np.mean(val_rmses):.4f}"
#             )
#             print(
#                 f"NMSE: train={np.mean(train_nmses):.4f}, val={np.mean(val_nmses):.4f}"
#             )

#         return self.fold_stats

#############################
####### DEPRECATED ##########
#############################


class DANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
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
    def __init__(
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
#
# def cross_validate_dann(
#     X,
#     Y,
#     tensor_dataset,
#     num_domains=5,
#     lambda_grl=0.3,
#     max_epochs=50,
#     patience=20,
#     batch_size=512,
#     gamma=0.1,
#     verbose=1,  # 0: silent, 1: plot + stats, 2: per-epoch logs
# ):
#     fold_stats = {}
#     train_rmses, val_rmses = [], []
#     train_nmses, val_nmses = [], []

#     for val_session in range(X.shape[0]):
#         train_sessions = [s for s in range(X.shape[0]) if s != val_session]

#         if verbose in [1, 2]:
#             print(
#                 f"\n=== Fold {val_session + 1} | Train on {train_sessions}, Validate on {val_session} ==="
#             )

#         session_ids_train = np.concatenate(
#             [np.full(X[s].shape[0], train_sessions.index(s)) for s in train_sessions]
#         )
#         train_dataset = tensor_dataset(
#             X[train_sessions], Y[train_sessions], session_ids_train
#         )
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#         X_val = X[val_session]
#         Y_val = Y[val_session]
#         session_ids_val = np.full(X_val.shape[0], val_session)
#         val_dataset = tensor_dataset(X_val, Y_val, session_ids_val)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         model = DANNModel(
#             lambda_grl=lambda_grl, num_domains=num_domains, output_dim=Y.shape[-1]
#         )
#         trainer = DANNTrainer(
#             model=model,
#             X=X,
#             Y=Y,
#             train_sessions=train_sessions,
#             val_session=val_session,
#             lambda_grl=lambda_grl,
#             gamma_entropy=gamma,
#             max_epochs=max_epochs,
#             patience=patience,
#             batch_size=batch_size,
#             tensor_dataset=tensor_dataset,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             verbose=(verbose == 2),
#         )

#         best_val_rmse = trainer.train()
#         trainer.model.eval()
#         train_preds = trainer.predict(train_loader)
#         train_rmse = trainer.train_losses[-1]
#         train_nmse = trainer.train_nmses[-1]

#         val_preds = trainer.predict(val_loader)
#         val_true = np.vstack([y for _, y, _ in val_loader.dataset])
#         val_rmse = RMSE(val_preds, val_true)
#         val_nmse = NMSE(val_preds, val_true)

#         if verbose in [1, 2]:
#             print(f"\nFold {val_session + 1}")
#             print(f"RMSE: train={train_rmse:.4f}, val={val_rmse:.4f}")
#             print(f"NMSE: train={train_nmse:.4f}, val={val_nmse:.4f}")

#         if verbose == 1:
#             fig = plt.figure()
#             plt.plot(trainer.train_losses, label="Training", color="blue", marker="s")
#             plt.plot(trainer.val_losses, label="Validation", color="red", marker="o")
#             plt.title(f"Average batch losses per epoch - Fold {val_session + 1}")
#             plt.xlabel("Epoch")
#             plt.ylabel("Loss")
#             plt.grid(True)
#             plt.legend(
#                 handles=[
#                     Line2D([0], [0], color="red", marker="o", linestyle="-"),
#                     Line2D([0], [0], color="blue", marker="s", linestyle="-"),
#                 ],
#                 labels=["Validation", "Training"],
#                 title="Groups",
#             )
#             plt.show()

#         train_rmses.append(train_rmse)
#         val_rmses.append(val_rmse)
#         train_nmses.append(train_nmse)
#         val_nmses.append(val_nmse)

#         fold_stats[f"Fold {val_session + 1}"] = {
#             "RMSE": {"train": train_rmse, "val": val_rmse},
#             "NMSE": {"train": train_nmse, "val": val_nmse},
#         }

#     # Averages
#     avg_rmse_train = np.mean(train_rmses)
#     avg_rmse_val = np.mean(val_rmses)
#     avg_nmse_train = np.mean(train_nmses)
#     avg_nmse_val = np.mean(val_nmses)

#     if verbose in [1, 2]:
#         print("\nAverage Scores across folds:")
#         print(f"RMSE: train={avg_rmse_train:.4f}, val={avg_rmse_val:.4f}")
#         print(f"NMSE: train={avg_nmse_train:.4f}, val={avg_nmse_val:.4f}")

#     fold_stats["Average"] = {
#         "RMSE": {"train": avg_rmse_train, "val": avg_rmse_val},
#         "NMSE": {"train": avg_nmse_train, "val": avg_nmse_val},
#     }

#     return fold_stats


# PREVIOUS ONE - KEPT JUST IN CASE EMERGENCY - WIP BEFORE BETTER CROSS VALIDATION
def cross_validate_dann(
    model_builder,
    X_folds,
    Y_folds,
    tensor_dataset,
    metric_fns,
    n_folds=4,
    batch_size=512,
    lambda_grl=0.3,
    max_epochs=50,
    patience=10,
    gamma=0.1,
    learning_rate=1e-3,
    device="cuda",
    verbose=0,
):
    results = {}

    for fold in range(n_folds):
        if verbose == 3:
            print(f"FOLD {fold + 1}/{n_folds}")
            fig = plt.figure()
            plt.title(f"Average batch losses per epoch - Fold {fold + 1}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend(
                handles=[
                    Line2D([0], [0], color="red", marker="o", linestyle="-"),
                    Line2D([0], [0], color="blue", marker="s", linestyle="-"),
                ],
                labels=["Validation", "Training"],
                title="Groups",
            )

        train_idx = list(range(n_folds))
        train_idx.remove(fold)
        val_idx = fold

        # Get train/val splits in legacy session format
        X_train = X_folds[train_idx]
        Y_train = Y_folds[train_idx]
        X_val = X_folds[val_idx]
        Y_val = Y_folds[val_idx]

        session_ids_train = np.concatenate(
            [np.full(X_folds[i].shape[0], i if i < fold else i - 1) for i in train_idx]
        )
        session_ids_val = np.full(X_val.shape[0], val_idx)

        train_dataset = tensor_dataset(X_train, Y_train, session_ids_train)
        val_dataset = tensor_dataset(X_val, Y_val, session_ids_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = model_builder()

        trainer = DANNTrainer(
            model=model,
            X=X_folds,
            Y=Y_folds,
            train_sessions=train_idx,
            val_session=val_idx,
            lambda_grl=lambda_grl,
            gamma_entropy=gamma,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            tensor_dataset=tensor_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=(verbose == 2),
        )

        logs = trainer.train()

        if verbose == 3:
            plt.plot(logs["epoch"], logs["Train RMSE"], marker="s", color="blue")
            plt.plot(logs["epoch"], logs["Val RMSE"], marker="o", color="red")
            plt.tight_layout()
            plt.show()

        Y_train_pred = trainer.predict(train_loader)
        Y_val_pred = trainer.predict(val_loader)
        Y_train_true = np.vstack([y for _, y, _ in train_loader.dataset])
        Y_val_true = np.vstack([y for _, y, _ in val_loader.dataset])

        results[fold] = {}
        for name, fn in metric_fns.items():
            results[fold][f"train_{name}"] = fn(Y_train_pred, Y_train_true)
            results[fold][f"val_{name}"] = fn(Y_val_pred, Y_val_true)

        if verbose == 2:
            print(f"\nFold {fold + 1}")
            for name in metric_fns:
                print(
                    f"{name}: train={results[fold][f'train_{name}']:.4f}, val={results[fold][f'val_{name}']:.4f}"
                )

    # Aggregate averages
    for name in metric_fns:
        train_vals = [results[fold][f"train_{name}"] for fold in range(n_folds)]
        val_vals = [results[fold][f"val_{name}"] for fold in range(n_folds)]
        results[f"avg_train_{name}"] = np.mean(train_vals)
        results[f"avg_val_{name}"] = np.mean(val_vals)

    if verbose >= 1:
        print("\nAverage Scores across folds:")
        for name in metric_fns:
            print(
                f"{name}: train={results[f'avg_train_{name}']:.4f}, val={results[f'avg_val_{name}']:.4f}"
            )

    return results


# def cross_validate_tempdann(
#     X,
#     Y,
#     tensor_dataset,
#     num_domains=5,
#     lambda_grl=0.3,
#     max_epochs=50,
#     patience=20,
#     batch_size=512,
#     gamma=0.1,
#     summary=False,
# ):
#     rmse_scores = []

#     for val_session in range(X.shape[0]):
#         train_sessions = [s for s in range(X.shape[0]) if s != val_session]
#         print(
#             f"\n=== Fold {val_session + 1} | Train on {train_sessions}, Validate on {val_session} ==="
#         )

#         # === Build train dataset
#         train_X = np.concatenate([X[s] for s in train_sessions], axis=0)
#         train_Y = np.concatenate([Y[s] for s in train_sessions], axis=0)
#         train_session_ids = np.concatenate(
#             [np.full(X[s].shape[0], train_sessions.index(s)) for s in train_sessions]
#         )

#         train_dataset = tensor_dataset(train_X, train_Y, train_session_ids)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#         # === Build validation dataset
#         val_X = X[val_session]
#         val_Y = Y[val_session]
#         val_session_ids = np.full(
#             val_X.shape[0], val_session
#         )  # Or a consistent identifier
#         val_dataset = tensor_dataset(val_X, val_Y, val_session_ids)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         model = TemporalDANNModel(
#             lambda_grl=lambda_grl,
#             num_domains=num_domains,
#             output_dim=Y.shape[-1],
#         )  # Instantiate TemporalDANNModel
#         trainer = DANNTrainer(
#             model=model,
#             X=X,
#             Y=Y,
#             train_sessions=train_sessions,
#             val_session=val_session,
#             lambda_grl=lambda_grl,
#             gamma_entropy=gamma,
#             max_epochs=max_epochs,
#             patience=patience,
#             batch_size=batch_size,
#             tensor_dataset=tensor_dataset,
#             train_loader=train_loader,
#             val_loader=val_loader,
#         )

#         fold_rmse = trainer.train()
#         rmse_scores.update(fold_rmse)

#         if summary:
#             plt.figure(figsize=(8, 5))
#             plt.plot(
#                 rmse_scores["epoch"],
#                 rmse_scores["Train RMSE"],
#                 label="Train",
#                 linestyle="--",
#                 marker="s",
#             )
#             plt.plot(
#                 rmse_scores["epoch"],
#                 rmse_scores["Val RMSE"],
#                 label="Val",
#                 linestyle="-",
#                 marker="o",
#             )
#             plt.xlabel("Epoch")
#             plt.ylabel("RMSE")
#             plt.title(f"Train/Val RMSE - {val_session}")
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

#     # best_val_rmse_per_fold = [min(log["Val RMSE"]) for log in rmse_scores.values()]
#     # mean_rmse = np.mean(best_val_rmse_per_fold)
#     # std_rmse = np.std(best_val_rmse_per_fold)

#     # print(f"\n=== Cross-validated RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} ===")


#     return rmse_scores