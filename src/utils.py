# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from sklearn.datasets import load_digits

# Libraries for modifying data
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate

# Libraries for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from typing import Sequence, Iterable, Optional, Tuple


def add_gaussian_noise(images, noise_level=0.5):
    """
    Adds Gaussian noise to images from the digits dataset.

    Args:
        images (np.ndarray): Array of shape (n_images, 8, 8) containing the images.
        noise_level (float): Standard deviation of the added Gaussian noise (default is 0.5).

    Returns:
        np.ndarray: New noisy images with values between 0 and 16.
    """
    noisy_images = images + noise_level * np.random.randn(*images.shape)
    
    # Clamp values to the [0, 16] range
    noisy_images = np.clip(noisy_images, 0, 16)
    
    return noisy_images



def add_impulse_noise(images, noise_ratio=0.05):
    """
    Adds impulse noise (salt & pepper) to the images.

    Args:
        images (np.ndarray): Array of images (n_images, 8, 8).
        noise_ratio (float): Proportion of pixels to corrupt (e.g., 0.05 = 5%).

    Returns:
        np.ndarray: Images with added impulse noise.
    """
    noisy_images = images.copy()
    n_images, h, w = images.shape
    n_total_pixels = h * w

    for idx in range(n_images):
        image = noisy_images[idx]
        n_noisy_pixels = int(noise_ratio * n_total_pixels)

        # Random noise positions
        coords = np.random.choice(n_total_pixels, n_noisy_pixels, replace=False)
        for c in coords:
            i, j = divmod(c, w)
            image[i, j] = np.random.choice([0, 16])  # salt or pepper

    return noisy_images



def visualize_transform(images, transform, index=0, title = 'Transformed'):
    """
    Visualizes the original and transformed images side by side.

    Args:
        images (np.ndarray): Array of shape (n_images, 8, 8).
        transform (function): Transformation function to apply to the images.
    """
    transformed_images = transform(images)


    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(images[index], cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(transformed_images[index], cmap='gray')
    axs[1].set_title(title)
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


class AddGaussianNoise:
    def __init__(self, noise_level=0.5):
        self.noise_level = noise_level

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Image tensor of shape (1, 28, 28) with values in [0, 1]
        Returns:
            torch.Tensor: Noisy image tensor, still in [0, 1]
        """
        noise = torch.randn(tensor.size()) * self.noise_level
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)
    
    

class AddImpulseNoise:
    def __init__(self, probability=0.05):
        """
        Args:
            probability (float): Proportion of pixels to be replaced by 0 or 1 (default = 5%)
        """
        self.probability = probability

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Image tensor of shape (1, 28, 28) with values in [0, 1]
        Returns:
            torch.Tensor: Noisy image tensor with impulse noise
        """
        noisy_tensor = tensor.clone()
        mask = torch.rand_like(tensor)

        # Apply salt (1) where mask < p/2, and pepper (0) where mask > 1 - p/2
        noisy_tensor[mask < self.probability / 2] = 0.0
        noisy_tensor[mask > 1 - self.probability / 2] = 1.0

        return noisy_tensor
    
def plot_validation_accuracy(x, y, x_label, title, color='C0', label='Validation Accuracy', convergence_epoch=-1):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color=color, label=label)
    if convergence_epoch != -1 and 0 <= convergence_epoch < len(x):
        plt.axvline(x=x[convergence_epoch], color='red', linestyle='--', label='Convergence Point')
    plt.ylim(0, 1)
    plt.xlabel(x_label)
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_multi_curves(
    x: Sequence[float],
    *y_vars: Iterable[float],
    labels: Optional[Sequence[str]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    markers: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = True,
    legend_loc: str = "best",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot several curves that all share the same x-axis.

    Parameters
    ----------
    x : 1-D sequence
        Common x-axis values.
    *y_vars : 1-D sequences
        Any number of y-series (each must have the same length as `x`).
        Example:   plot_multi_curves(x, y1, y2, y3, labels=[...])
                   plot_multi_curves(x, *[y1, y2], labels=[...])
    labels : sequence of str, optional
        One label per y-series. Required if you want a legend.
    title, xlabel, ylabel : str, optional
        Figure and axis labels.
    markers, linestyles : sequences, optional
        Custom marker and line styles (same length as *y_vars*).
        Defaults: Matplotlib’s cycling styles.
    figsize : tuple of int, optional
        Figure dimensions in inches.
    grid : bool, optional
        Toggle grid lines.
    legend_loc : str, optional
        Legend position (same `loc` argument as `plt.legend`).
    save_path : str, optional
        If not None, save the figure to this file (png, pdf, …).

    Raises
    ------
    ValueError
        If lengths of inputs are inconsistent.

    Examples
    --------
    >>> # Suppose you already defined the following arrays:
    >>> # num_examples_MNIST_Monte_Carlo
    >>> # num_examples_MNIST_Gaussian_cum_curr_learn
    >>> x_points = [1, 2, 3, 4, 5]
    >>> plot_multi_curves(
    ...     x_points,
    ...     num_examples_MNIST_Monte_Carlo,
    ...     num_examples_MNIST_Gaussian_cum_curr_learn,
    ...     labels=["Monte-Carlo", "Gaussian + Curriculum"],
    ...     title="MNIST – Validation Accuracy vs Training Size",
    ...     xlabel="Training samples",
    ...     ylabel="Validation Accuracy",
    ...     markers=["o", "s"],
    ...     linestyles=["-", "--"],
    ... )
    """
    # --- basic checks --------------------------------------------------------
    if not y_vars:
        raise ValueError("At least one y-series must be provided.")

    if labels is not None and len(labels) != len(y_vars):
        raise ValueError("`labels` must match the number of y-series.")

    if markers is not None and len(markers) != len(y_vars):
        raise ValueError("`markers` must match the number of y-series.")

    if linestyles is not None and len(linestyles) != len(y_vars):
        raise ValueError("`linestyles` must match the number of y-series.")

    x = np.asarray(x)
    plt.figure(figsize=figsize)

    # --- draw each curve -----------------------------------------------------
    for idx, y in enumerate(y_vars):
        y_arr = np.asarray(y)
        if y_arr.shape != x.shape:
            raise ValueError(
                f"y-series #{idx} length ({y_arr.size}) does not match x ({x.size})"
            )

        marker = markers[idx] if markers else None
        linestyle = linestyles[idx] if linestyles else "-"
        label = labels[idx] if labels else None

        plt.plot(x, y_arr, marker=marker, linestyle=linestyle, label=label)

    # --- aesthetics ----------------------------------------------------------
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if grid:
        plt.grid(alpha=0.3)

    if labels:  # only draw legend when labels are supplied
        plt.legend(loc=legend_loc)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
