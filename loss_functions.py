import torch
from torch import nn
    
class NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()

    def forward(self, y_pred, y_true):
        num = (y_true - y_pred).pow(2).sum(dim=1).mean()
        den = (y_true - y_true.mean(dim=0)).pow(2).sum(dim=1).mean()
        return num / den

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        return torch.sqrt(mse_loss)