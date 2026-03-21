import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch


def plot_distribution(data, bin=100):

    data.plot.hist(bins=bin)
    plt.show()

def plot_line(csv_file, save_path, x = 'Epoch', y = ['Train Loss', 'Val Loss'], ylim=None):

    data = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = y
    for i, metric in enumerate(metrics):
        ax.plot(
            data[x],
            data[metric],
            markersize=5,
            linestyle='-',
            linewidth=2,
            label=metric
        )


    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('loss or error', fontsize=12)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.autoscale(enable=True, axis='y', tight=True)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_scatter(x_tensor, y_tensor, save_path, xlim=None, ylim=None):

    if isinstance(x_tensor, torch.Tensor):
        x_tensor = x_tensor.detach().cpu().numpy()
    if isinstance(y_tensor, torch.Tensor):
        y_tensor = y_tensor.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_tensor, y_tensor, alpha=0.8, edgecolors='k', label="Data Points")

    if xlim is None:
        xmin, xmax = np.min(x_tensor), np.max(x_tensor)
        xlim = (xmin, xmax)
    if ylim is None:
        ymin, ymax = np.min(y_tensor), np.max(y_tensor)
        ylim = (ymin, ymax)

    plt.xlim(xlim)
    plt.ylim(ylim)

    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="y = x")

    plt.xlabel('Target Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
