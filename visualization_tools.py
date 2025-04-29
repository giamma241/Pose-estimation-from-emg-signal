import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_time_window(time_window, low=-2000, high=2000):
    """
    Visualize the eight EMG signals in a given time window.

    Args:
        time_window (np.ndarray): Array of shape (8, size), where 8 is the number of EMG sensors
                                  and size is the number of time samples.
        low (float, optional): Minimum y-axis limit for the plots. Default is -2000.
        high (float, optional): Maximum y-axis limit for the plots. Default is 2000.

    Behavior:
        - Creates a 4x2 grid of subplots (one for each EMG sensor).
        - Plots the signal of each sensor separately.
        - Sets consistent y-axis limits (low, high) for better comparison.
        - Adds grids and titles for clarity.
    """
    size = time_window.shape[1]

    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axs_flat = axs.flatten()

    for i in range(8):
        axs_flat[i].plot(np.arange(size), time_window[i])
        axs_flat[i].set_title(f'Sensor {i}')
        axs_flat[i].set_ylim(low,high)
        axs_flat[i].grid(True)

def scatter_3d_points(data, ax = None, color = 'b', alpha = 1):
    """
    Visualizes a 3d scatterplot of a bunch of points in R3

    Args:
        data (np.ndarray): a numpy array of shape (N, 3)
    """
    X = data[:,[0]]
    Y = data[:,[1]]
    Z = data[:,[2]]

    if not ax:
        fig = plt.figure(constrained_layout=True)
        ax  = fig.add_subplot(projection='3d')
    plt.ion()
    ax.scatter(X,Y,Z,marker='.',color=color, alpha = alpha)
    ax.set_xlabel('Roll')
    ax.set_ylabel('Yaw')
    ax.set_zlabel('Pitch')

    return ax