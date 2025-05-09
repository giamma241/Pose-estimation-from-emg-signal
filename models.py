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
