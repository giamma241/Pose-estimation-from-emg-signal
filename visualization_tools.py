import numpy as np
import matplotlib.pyplot as plt

def visualize_time_window(time_window, low=-2000, high=2000):
    """
    visualizes the eight signals of a time window
    time_window is a np array of shape (8, size) where size is the time window size 
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