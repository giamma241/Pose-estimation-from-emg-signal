import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class TrainingManager:
    """Helper class which given a model and hyperparameters will train the model."""

    def __init__(self, model, train_loader, val_loader, training_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.val_rmses = []

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        criterion = nn.MSELoss()

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(self.train_loader.dataset)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            val_rmse = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    rmse = torch.sqrt(torch.mean((outputs - y_batch) ** 2))
                    val_loss += loss.item() * X_batch.size(0)
                    val_rmse += rmse.item() * X_batch.size(0)

            val_loss /= len(self.val_loader.dataset)
            val_rmse /= len(self.val_loader.dataset)
            self.val_losses.append(val_loss)
            self.val_rmses.append(val_rmse)

            if (epoch + 1) % self.config.get("log_every", 1) == 0:
                print(
                    f"Epoch {epoch + 1:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}"
                )

    def get_logs(self):
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_rmses": self.val_rmses,
        }


class CrossValidationManager:
    """
    Helper class to execute a k-fold cross validation using the TrainingManager class and hyperparameters.
    By default set to 4 folds, may be adjusted.
    """

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
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.data = data
        self.labels = labels
        self.training_config = training_config
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config or {}
        self.n_folds = n_folds

    def run(self):
        experiment_log = {
            "architecture": self.model_config,
            "training_config": self.training_config,
            "folds": [],
        }

        for fold_idx in range(self.n_folds):
            print(f"\n===== Fold {fold_idx + 1}/{self.n_folds} =====")

            X_train = np.vstack(
                [self.data[i] for i in range(self.n_folds) if i != fold_idx]
            )
            y_train = np.vstack(
                [self.labels[i] for i in range(self.n_folds) if i != fold_idx]
            )

            X_val = self.data[fold_idx]
            y_val = self.labels[fold_idx]

            train_dataset = self.dataset_class(X_train, y_train, **self.dataset_config)
            val_dataset = self.dataset_class(X_val, y_val, **self.dataset_config)

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
            model.build()

            trainer = TrainingManager(
                model, train_loader, val_loader, self.training_config
            )
            trainer.train()

            fold_log = {"fold_number": fold_idx, "metrics": trainer.get_logs()}
            experiment_log["folds"].append(fold_log)

        return experiment_log


class EMGConvNet(nn.Module):
    def __init__(
        self,
        conv_layers_config,  # list of (out_channels, kernel_size, stride)
        fc_layers_config,  # list of fully connected layer sizes
        output_dim=51,
        conv_dropouts=None,  # optional list of dropout probs for conv layers
        fc_dropouts=None,  # optional list of dropout probs for fc layers
        verbose=False,
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
                    nn.ReLU(),
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
        conv_layers_config,  # list of (out_channels, kernel_height, kernel_width, stride)
        fc_layers_config,  # list of fully connected layer sizes
        output_dim=51,
        conv_dropouts=None,  # list of dropout probabilities per conv layer
        fc_dropouts=None,  # list of dropout probabilities per fc layer
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.conv_layers = nn.ModuleList()
        self.conv_dropouts = conv_dropouts or [0.0] * len(conv_layers_config)

        in_channels = 1  # 1 input channel since we're reshaping (8, 500) to (1, 8, 500)

        for (out_channels, kh, kw, stride), dropout in zip(
            conv_layers_config, self.conv_dropouts
        ):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kh, kw),
                        stride=(1, stride),
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
        # x shape: (B, 8, 500) -> reshape to (B, 1, 8, 500)
        x = x.unsqueeze(1)
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
            dummy_input = torch.zeros(1, 8, 500)
            _ = self.forward(dummy_input)
