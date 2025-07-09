"""
Enhanced visualization tools for filter analysis.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse


def plot_covariance_ellipse(
    mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, n_std: float = 2.0, **kwargs
):
    """
    Plot covariance ellipse for 2D data.

    Args:
        mean: Mean vector (2D)
        cov: 2x2 covariance matrix
        ax: Matplotlib axis
        n_std: Number of standard deviations
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_adaptive_parameters(results: Dict[str, Any]):
    """
    Plot evolution of adaptive noise parameters.

    Args:
        results: Filter results dictionary
    """
    Q_history = np.array(results["Q_adaptive"])
    R_history = np.array(results["R_adaptive"])
    times = results["times"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Process noise diagonal elements
    ax1.semilogy(times, Q_history[:, 0, 0], label="Q[0,0] (x)")
    ax1.semilogy(times, Q_history[1, 1, 1], label="Q[1,1] (y)")
    ax1.semilogy(times, Q_history[2, 2, 2], label="Q[2,2] (z)")
    ax1.set_ylabel("Process Noise Variance")
    ax1.set_title("Adaptive Process Noise Evolution")
    ax1.legend()
    ax1.grid(True)

    # Measurement noise diagonal elements
    ax2.semilogy(times, R_history[:, 0, 0], label="R[0,0] (x)")
    ax2.semilogy(times, R_history[:, 1, 1], label="R[1,1] (y)")
    ax2.semilogy(times, R_history[:, 2, 2], label="R[2,2] (z)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Measurement Noise Variance")
    ax2.set_title("Adaptive Measurement Noise Evolution")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("adaptive_parameters.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_analysis(
    truth: np.ndarray, estimates: np.ndarray, covariances: np.ndarray, times: List
):
    """
    Plot detailed error analysis.

    Args:
        truth: True states
        estimates: Estimated states
        covariances: State covariances
        times: Time vector
    """
    errors = truth - estimates

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    labels = ["X", "Y", "Z", "Vx", "Vy", "Vz"]
    units = ["km", "km", "km", "km/s", "km/s", "km/s"]

    for i in range(6):
        ax = axes[i]

        # Plot error
        ax.plot(times, errors[:, i], "b-", alpha=0.7, label="Error")

        # Plot 3-sigma bounds
        three_sigma = 3 * np.sqrt(covariances[:, i, i])
        ax.fill_between(
            times, -three_sigma, three_sigma, alpha=0.3, color="gray", label="3σ bounds"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{labels[i]} Error ({units[i]})")
        ax.set_title(f"{labels[i]} Component Error")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("error_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
