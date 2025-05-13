import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

""" 
EXERPT PAPER ON CONCLUSIVE NN ARCHITECTURES

After trying multiple network architectures, the following
architecture achieved the best cross-validated performance both for
phasic and tonic EMG:
• A linear temporal convolution with 128 filters of
shape (1 × 16).
• A spatial convolution with 16 filters of shape (8 × 1) followed
by a batch normalization layer, a ReLU non-linear activation
function, a mean pooling layer of shape (1 × 4) and a spatial
dropout layer.
• A temporal convolution with 128 filters of shape (1 × 16)
followed by a batch normalization layer, a ReLU non-linear
activation function, a mean pooling layer of shape (1 × 4) and
a spatial dropout layer.
• Two fully-connected layers of 128 neurons each followed by a
ReLU non-linear activation function.
• A fully-connected layer of four neurons (corresponding to the
4 different gestures) followed by a softmax activation function. """

###################### ERROR FUNCTIONS
class NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true) -> torch.Tensor:
        num = (y_true - y_pred).pow(2).sum()
        y_true_mean = torch.mean(y_true, dim = 0)
        den = (y_true - y_true_mean).pow(2).sum() + self.eps
        return num / den

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        return torch.sqrt(mse_loss + self.eps)
##############################

class TrainingManager:
    def __init__(
        self, model, train_loader, val_loader, training_config, criterion=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.val_rmses = []
        self.val_nmses = []

    def __call__(self):
        self.train()

    def train(self):
        self.model.to(self.device)
        self._train_loop()

    def _train_loop(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])

        for epoch in range(self.config["epochs"]):
            train_loss = self._train_epoch(optimizer)
            val_loss, val_rmse, val_nmse = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_rmses.append(val_rmse)
            self.val_nmses.append(val_nmse)

            if (epoch + 1) % self.config.get("log_every", 1) == 0:
                print(
                    f"Epoch {epoch + 1:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val NMSE: {val_nmse:.4f}"
                )

    def _train_epoch(self, optimizer):
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_nmse = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                rmse = torch.sqrt(torch.mean((outputs - y_batch) ** 2))
                y_mean = torch.mean(y_batch, dim=0)
                nmse_num = torch.sum((y_batch - outputs) ** 2)
                nmse_den = torch.sum((y_batch - y_mean) ** 2) + 1e-8
                nmse = nmse_num / nmse_den

                val_loss += loss.item() * X_batch.size(0)
                val_rmse += rmse.item() * X_batch.size(0)
                val_nmse += nmse.item() * X_batch.size(0)

        N = len(self.val_loader.dataset)
        return val_loss / N, val_rmse / N, val_nmse / N

    def get_logs(self):
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_rmses": self.val_rmses,
            "val_nmses": self.val_nmses,
        }

    def get_validation_predictions(self):
        preds = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds.append(outputs.cpu())
                targets.append(y_batch)
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        return preds, targets


class SessionCrossValidator:
    def __init__(
        self,
        model_class,
        model_config,
        X_sessions,
        Y_sessions,
        training_config,
        dataset_class,
        dataset_config=None,
        save_predictions=False,
        criterion=None,
    ):
        assert len(X_sessions) == 4, "Expected exactly 4 sessions for CV"
        self.model_class = model_class
        self.model_config = model_config
        self.X_sessions = X_sessions
        self.Y_sessions = Y_sessions
        self.training_config = training_config
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config or {}
        self.save_predictions = save_predictions
        self.criterion = criterion

    def _standardise(self, X_train, X_val):
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        return (X_train - mean) / std, (X_val - mean) / std

    def run(self):
        logs = []
        for fold_idx in range(4):
            print(f"\n===== Fold {fold_idx + 1}/4 (session {fold_idx} as val) =====")

            X_val = self.X_sessions[fold_idx]
            Y_val = self.Y_sessions[fold_idx]
            X_train = np.vstack([self.X_sessions[i] for i in range(4) if i != fold_idx])
            Y_train = np.vstack([self.Y_sessions[i] for i in range(4) if i != fold_idx])

            X_train, X_val = self._standardise(X_train, X_val)

            train_dataset = self.dataset_class(X_train, Y_train, **self.dataset_config)
            val_dataset = self.dataset_class(X_val, Y_val, **self.dataset_config)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config["batch_size"],
                shuffle=False,
            )

            model = self.model_class(**self.model_config)
            trainer = TrainingManager(
                model,
                train_loader,
                val_loader,
                self.training_config,
                criterion=self.criterion,
            )
            trainer.train()

            fold_log = {"fold": fold_idx, "metrics": trainer.get_logs()}
            if self.save_predictions:
                preds, targets = trainer.get_validation_predictions()
                fold_log["predictions"] = preds.tolist()
                fold_log["targets"] = targets.tolist()

            logs.append(fold_log)
        return logs


### LIKELY WRONG AND CAUSING DATA LEAKAGE BY FLATTENING INPUT DATA
class CrossValidationManager:
    def __init__(
        self,
        model_class,
        model_config,
        data,
        labels,
        training_config,
        dataset_class,
        dataset_config=None,
        n_folds=4,
        save_predictions=False,
        criterion=None,
        is_sklearn_model=False,
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.data = data
        self.labels = labels
        self.training_config = training_config
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config or {}
        self.n_folds = n_folds
        self.save_predictions = save_predictions
        self.criterion = criterion
        self.is_sklearn_model = is_sklearn_model

    def _prepare_fold_data(self, fold_idx):
        X_train = np.vstack(
            [self.data[i] for i in range(self.n_folds) if i != fold_idx]
        )
        y_train = np.vstack(
            [self.labels[i] for i in range(self.n_folds) if i != fold_idx]
        )
        X_val = self.data[fold_idx]
        y_val = self.labels[fold_idx]
        return X_train, y_train, X_val, y_val

    def _train_sklearn_model(self, X_train, y_train, X_val, y_val, fold_idx):
        model = self.model_class(**self.model_config)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        print(f"Sklearn Model Fold {fold_idx + 1} RMSE: {rmse:.4f}")
        fold_log = {"fold_number": fold_idx, "rmse": rmse}
        if self.save_predictions:
            fold_log["predictions"] = predictions.tolist()
            fold_log["targets"] = y_val.tolist()
        return fold_log

    def _train_torch_model(self, X_train, y_train, X_val, y_val, fold_idx):
        train_dataset = self.dataset_class(X_train, y_train, **self.dataset_config)
        val_dataset = self.dataset_class(X_val, y_val, **self.dataset_config)
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.training_config["batch_size"], shuffle=False
        )
        model = self.model_class(**self.model_config)
        if hasattr(model, "build"):
            model.build()
        trainer = TrainingManager(
            model,
            train_loader,
            val_loader,
            self.training_config,
            criterion=self.criterion,
        )
        trainer.train()
        fold_log = {"fold_number": fold_idx, "metrics": trainer.get_logs()}
        if self.save_predictions:
            predictions, targets = trainer.get_validation_predictions()
            fold_log["predictions"] = predictions.tolist()
            fold_log["targets"] = targets.tolist()
        return fold_log

    def run(self):
        experiment_log = {
            "architecture": self.model_config,
            "training_config": self.training_config,
            "folds": [],
        }

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.data)):
            print(f"\n===== Fold {fold_idx + 1}/{self.n_folds} =====")
            X_train, X_val = self.data[train_idx], self.data[val_idx]
            y_train, y_val = self.labels[train_idx], self.labels[val_idx]

            if self.is_sklearn_model:
                fold_log = self._train_sklearn_model(
                    X_train, y_train, X_val, y_val, fold_idx
                )
            else:
                fold_log = self._train_torch_model(
                    X_train, y_train, X_val, y_val, fold_idx
                )

            experiment_log["folds"].append(fold_log)

        return experiment_log


##### NEURAL NETWORKS
class ConvNN(nn.Module):
    def __init__(self, end_dim):
        super().__init__()

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
        pass # here for compatibility

class EMGConvNet(nn.Module):
    def __init__(
        self,
        conv_layers_config,  # list of (out_channels, kernel_size, stride)
        fc_layers_config,  # list of fully connected layer sizes
        output_dim=51,
        conv_dropouts=None,  # optional list of dropout probs for conv layers
        fc_dropouts=None,  # optional list of dropout probs for fc layers
        verbose=False,
        activation_function=nn.ReLU(),  # activation function toggle
    ):
        super().__init__()
        self.verbose = verbose

        self.conv_layers = nn.ModuleList()
        self.conv_dropouts = conv_dropouts or [0.0] * len(conv_layers_config)
        in_channels = 8  # EMG input channels

        # --- Build convolutional blocks
        for idx, ((out_channels, kernel_size, stride), dropout_rate) in enumerate(
            zip(conv_layers_config, self.conv_dropouts)
        ):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride),
                    nn.BatchNorm1d(out_channels),  # Batch Normalisation per channel
                    activation_function,
                    nn.Dropout(p=dropout_rate),
                )
            )
            in_channels = out_channels

        self.flattened_size = None
        self._fc_config = fc_layers_config
        self.fc_dropouts = fc_dropouts or [0.0] * len(fc_layers_config)
        self.fc_layers = nn.ModuleList()
        self._output_layer = None
        self.output_dim = output_dim

    def forward(self, x):
        if self.verbose:
            print(f"Input shape: {x.shape}")  # (B, 8, 500)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.verbose:
                print(f"After conv layer {i}: {x.shape}")

        B, C, T = x.shape
        x = x.view(B, -1)

        if self.flattened_size is None:
            self.flattened_size = x.shape[1]
            self._build_fc_layers()

        if self.verbose:
            print(f"After flattening: {x.shape}")

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if self.verbose:
                print(f"After FC layer {i}: {x.shape}")

        x = self._output_layer(x)
        if self.verbose:
            print(f"Final output: {x.shape}")
        return x

    def _build_fc_layers(self):
        in_features = self.flattened_size
        for hidden_dim, dropout in zip(self._fc_config, self.fc_dropouts):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)
                )
            )
            in_features = hidden_dim
        self._output_layer = nn.Linear(in_features, self.output_dim)

    def build(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 8, 500)
            _ = self.forward(dummy_input)

class EMGConvNet2D(nn.Module):
    def __init__(
        self,
        input_channels,  # <- This is the number of features: e.g., 8 or 24
        conv_layers_config,  # list of (out_channels, kernel_height, kernel_width, stride_height, stride_width)
        fc_layers_config,  # list of fully connected layer sizes
        output_dim=51,
        conv_dropouts=None,
        fc_dropouts=None,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.input_channels = (
            input_channels  # <--- number of input feature channels (raw + deltas)
        )

        self.conv_layers = nn.ModuleList()
        self.conv_dropouts = conv_dropouts or [0.0] * len(conv_layers_config)
        in_channels = 1  # we're always reshaping input to (B, 1, C, T)

        for (out_channels, kh, kw, sh, sw), dropout in zip(
            conv_layers_config, self.conv_dropouts
        ):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kh, kw),
                        stride=(sh, sw),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout2d(p=dropout),
                )
            )
            in_channels = out_channels

        self.flattened_size = None
        self._fc_config = fc_layers_config
        self.fc_dropouts = fc_dropouts or [0.0] * len(fc_layers_config)
        self.fc_layers = nn.ModuleList()
        self._output_layer = None
        self.output_dim = output_dim

    def forward(self, x):
        # x: (B, C, T) → reshape to (B, 1, C, T)
        x = x.view(x.size(0), 1, self.input_channels, x.size(2))

        if self.verbose:
            print(f"Input reshaped to: {x.shape}")

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.verbose:
                print(f"After conv layer {i}: {x.shape}")

        x = x.view(x.size(0), -1)

        if self.flattened_size is None:
            self.flattened_size = x.shape[1]
            self._build_fc_layers()

        if self.verbose:
            print(f"After flattening: {x.shape}")

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if self.verbose:
                print(f"After FC layer {i}: {x.shape}")

        x = self._output_layer(x)
        if self.verbose:
            print(f"Final output: {x.shape}")

        return x

    def _build_fc_layers(self):
        in_features = self.flattened_size
        for hidden_dim, dropout in zip(self._fc_config, self.fc_dropouts):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)
                )
            )
            in_features = hidden_dim
        self._output_layer = nn.Linear(in_features, self.output_dim)

    def build(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 500)
            self.forward(dummy_input)


### DANN - Domain Adversarial Neural Network
## Improves generalisation - tdb: feed session specific info to better identify session specific infos.


import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset


# ======= Dataset =======
class EMGWindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.reshape(
            -1, *X.shape[2:]
        )  # (n_sessions * n_windows, channels, window)
        self.Y = Y.reshape(-1, Y.shape[-1])
        self.session_ids = torch.arange(X.shape[0]).repeat_interleave(X.shape[1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            self.session_ids[idx],
        )


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
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
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


# ======= Training Loop =======
def train_dann(
    model, dataloader, optimizer, reg_loss_fn, dom_loss_fn, lambda_grl, device="gpu"
):
    model.train()
    for x, y, s in dataloader:
        x, y, s = x.to(device), y.to(device), s.to(device)

        y_pred, domain_pred = model(x)
        loss_reg = reg_loss_fn(y_pred, y)
        loss_dom = dom_loss_fn(domain_pred, s)
        loss_total = loss_reg + lambda_grl * loss_dom

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    print(f"Reg Loss: {loss_reg.item():.4f}, Dom Loss: {loss_dom.item():.4f}")


# ====== Training tools ====


# === Dataset that tracks session IDs ===
class DANNWindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.reshape(-1, *X.shape[2:])  # (N, 8, 500)
        self.Y = Y.reshape(-1, Y.shape[-1])  # (N, 51)
        self.session_ids = torch.arange(X.shape[0]).repeat_interleave(X.shape[1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            self.session_ids[idx],
        )


# === Training class ===
class DANNTrainer:
    def __init__(
        self,
        model: nn.Module,
        X: np.ndarray,
        Y: np.ndarray,
        train_sessions: list,
        val_session: int,
        lambda_grl: float = 0.1,
        gamma_entropy=0.5,
        batch_size: int = 128,
        max_epochs: int = 50,
        patience: int = 10,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tensor_dataset=DANNWindowDataset,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.lambda_grl = lambda_grl
        self.gamma_entropy = gamma_entropy
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                     self.optimizer,
        #                     mode='min',
        #                     factor=0.5,
        #                     patience=5,         # wait 5 epochs with no improvement
        #                     threshold=1e-4,     # only trigger if change is meaningful
        #                     )
        self.reg_loss = nn.MSELoss()
        self.dom_loss = nn.CrossEntropyLoss()

        # Data
        full_dataset = tensor_dataset(X, Y)
        sid = full_dataset.session_ids.numpy()
        self.train_loader = DataLoader(
            torch.utils.data.Subset(
                full_dataset, np.where(np.isin(sid, train_sessions))[0]
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, np.where(sid == val_session)[0]),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def entropy_loss(self, logits):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)  # avoid log(0)
        return -torch.sum(probs * log_probs, dim=1).mean()

    def train(self):
        best_val_rmse = float("inf")
        best_weights = None
        patience_counter = 0

        def compute_lambda(epoch, max_lambda=1.0, warmup_epochs=10):
            return min(max_lambda, epoch / warmup_epochs)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_losses = []

            lambda_grl = compute_lambda(epoch, max_lambda=1.0, warmup_epochs=10)

            for x, y, sid in self.train_loader:  # ✅ must be self.train_loader
                x, y, sid = x.to(self.device), y.to(self.device), sid.to(self.device)

                y_pred, dom_pred = self.model(
                    x, lambda_grl=lambda_grl
                )  # ✅ call on self.model

                loss = (
                    self.reg_loss(y_pred, y)
                    + lambda_grl * self.dom_loss(dom_pred, sid)
                    - self.gamma_entropy * self.entropy_loss(dom_pred)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            val_rmse = self.evaluate()
            domain_train_acc = self.evaluate_domain_on_train()
            # self.scheduler.step(val_rmse)

            train_loss = np.mean(epoch_losses)  # ✅ needed for print
            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Dom Train Acc: {domain_train_acc:.2%}"
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

        self.model.load_state_dict(best_weights)
        return best_val_rmse

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                yp, _ = self.model(x)
                y_pred.append(yp.cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        return rmse

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

        s_true = np.concatenate(s_true)
        s_pred = np.concatenate(s_pred)
        domain_acc = (s_pred == s_true).mean()
        return domain_acc


def cross_validate_dann(
    X, Y, tensor_dataset, lambda_grl=0.3, max_epochs=50, patience=20, batch_size=512
):
    rmse_scores = []

    for val_session in range(X.shape[0]):
        train_sessions = [s for s in range(X.shape[0]) if s != val_session]
        print(
            f"\n=== Fold {val_session + 1} | Train on {train_sessions}, Validate on {val_session} ==="
        )

        model = DANNModel(lambda_grl=lambda_grl, num_domains=5, output_dim=Y.shape[-1])
        trainer = DANNTrainer(
            model=model,
            X=X,
            Y=Y,
            train_sessions=train_sessions,
            val_session=val_session,
            lambda_grl=lambda_grl,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            tensor_dataset=tensor_dataset,
        )

        fold_rmse = trainer.train()
        rmse_scores.append(fold_rmse)

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"\n=== Cross-validated RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} ===")

    return rmse_scores


# rmse_scores = cross_validate_dann(X_windows, Y_labels, tensor_dataset=DANNWindowDataset) <= Run this for cross validation.
