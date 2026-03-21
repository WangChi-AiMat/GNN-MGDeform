import numpy as np
import torch


def gaussian_angle_expansion(angles, num_gauss=8, gamma=10.0):

    if isinstance(angles, torch.Tensor):
        angles = angles.cpu().numpy()

    angles = np.array(angles, dtype=np.float32)

    if len(angles) == 0:
        return np.zeros(num_gauss, dtype=np.float32)

    # Gaussian centers (0 ~ π)
    centers = np.linspace(0, np.pi, num_gauss)

    # [num_angles, num_gauss]
    diff = angles.reshape(-1, 1) - centers.reshape(1, -1)

    # Gaussian basis for all angles
    gauss_vals = np.exp(-gamma * diff * diff)

    feature = gauss_vals.mean(axis=0)

    return feature.astype(np.float32)
