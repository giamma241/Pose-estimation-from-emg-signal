import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BEST_MODELS_PATH = 'best_models'

LATENT_DIMS = {
        0: 1, 1: 1, 2: 1,
        3: 1, 4: 2, 5: 1,
        6: 1, 7: 2, 8: 1,
        9: 1, 10: 2, 11: 1,
        12: 1, 13: 1, 14: 2,
        15: 1, 16: 1,}

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # encoder: [3 → 64 → 32 → 16 → latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, 4),
            nn.SELU(),
            nn.Linear(4, latent_dim)  # No activation
        )
        
        # decoder: [latent_dim → 16 → 32 → 64 → 3]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.SELU(),
            nn.Linear(4, 8),
            nn.SELU(),
            nn.Linear(8, 16),
            nn.SELU(),
            nn.Linear(16, 3)  # no activation
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # LeCun normal initialization for SELU
                nn.init.normal_(m.weight, 0, std=1. / np.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    

def trainer(
        Y,
        bone,
        latent_dim,
        best_models_path,
        batch_size = 4096,
        learning_rate = 1e-3,
        weight_decay = 1e-4,
        max_epochs = 100,
        patience = 20,
        num_workers = 4):
    # ----------------------------
    # 1. Configuration
    # ----------------------------
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.backends.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # ----------------------------
    # 2. Prepare Data (CPU‑only)
    # ----------------------------
    Y_bone = Y[:, bone*3:(bone+1)*3]

    Yb_cpu   = torch.as_tensor(Y_bone, dtype=torch.float32)  # stays on CPU

    dataset      = TensorDataset(Yb_cpu, Yb_cpu)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_loader   = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )

    # instantiate with your chosen latent dimension
    model = AutoEncoder(latent_dim).to(device)

    # ----------------------------
    # 4. Loss & Optimizer
    # ----------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # ----------------------------
    # 5. Training Loop w/ Early Stopping
    # ----------------------------
    best_val_loss     = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, max_epochs+1):
        # —— Training —— 
        model.train()
        running_loss = 0.0

        for xb_cpu, yb_cpu in train_loader:
            # move batch to MPS (or CUDA) on-the-fly
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(xb)              
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # —— Validation ——
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb_cpu, yb_cpu in val_loader:
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)
                running_val += criterion(model(xb), yb).item()
        avg_val_loss = running_val / len(val_loader)

        print(f"Epoch {epoch:02d}: Train MSE = {avg_train_loss:.6f} | Val MSE = {avg_val_loss:.6f}")

        # —— Early Stopping ——
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_models_path + f"/best_model_bone{bone}.pth")
            print("  → New best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break


def encode(Y, best_models_path, device = 'mps'):
    Y_shape = Y.shape
    
    Y_enc = []
    for bone in range(17):
        latent_dim = LATENT_DIMS[bone]
        model = AutoEncoder(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(best_models_path + f'/best_model_bone{bone}.pth'))
        
        to_encode = Y[..., 3*bone:3*(bone+1)].reshape(-1,3)
        to_encode_tensor = torch.tensor(
            to_encode,
            dtype=torch.float,
            device=device)
        encoded_tensor = model.encoder(to_encode_tensor)
        Y_enc.append(encoded_tensor.cpu().detach().numpy())

    Y_enc = np.hstack(Y_enc)
    Y_enc = Y_enc.reshape(*Y_shape[:-1], 21)
    
    return Y_enc


def decode(Y, best_models_path, device = 'mps'):
    Y_shape = Y.shape
    
    current = 0
    Y_dec = []
    for bone in range(17):
        latent_dim = LATENT_DIMS[bone]
        model = AutoEncoder(latent_dim=latent_dim).to(device)

        model.load_state_dict(torch.load(best_models_path + f'/best_model_bone{bone}.pth'))

        to_decode = Y[..., current:current + latent_dim].reshape(-1, latent_dim)
        current += latent_dim

        to_decode_tensor = torch.tensor(
            to_decode,
            dtype=torch.float,
            device=device)
        decoded_tensor = model.decoder(to_decode_tensor)
        Y_dec.append(decoded_tensor.cpu().detach().numpy())
        
    Y_dec = np.hstack(Y_dec)
    Y_dec = Y_dec.reshape(*Y_shape[:-1], 51)

    return Y_dec