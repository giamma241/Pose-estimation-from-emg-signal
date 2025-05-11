import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from torch.utils.data import TensorDataset, DataLoader

import torch
from torch import optim

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