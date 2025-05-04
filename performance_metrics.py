import numpy as np

def RMSE(y_pred, y_val):
    """
    Computes the root mean squared error of y_pred with respect to y_val
    assumes a shape (N, 51)
    """
    return (np.sum((y_val - y_pred)**2)/np.prod(y_val.shape))**0.5

def NMSE(y_pred, y_val):
    """
    Computes the normalized mean squared error of y_pred with respect to y_val
    assumes a shape (N, 51)
    """
    num = np.sum((y_val - y_pred)**2)
    den = np.sum((y_val - np.mean(y_val, axis=0))**2)
    return num/den