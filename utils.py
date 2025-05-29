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


def add_noise(images, noise_level=0.5):
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


def add_blur(images, sigma=1.0):
    """
    Applies Gaussian blur to the images.

    Args:
        images (np.ndarray): Array of shape (n_images, 8, 8).
        sigma (float): Degree of blur (the larger the sigma, the blurrier the image).

    Returns:
        np.ndarray: Blurred images.
    """
    blurred_images = np.array([gaussian_filter(img, sigma=sigma) for img in images])
    return np.clip(blurred_images, 0, 16)



def rotate_images(images, angle=15):
    """
    Rotates each image in the array around its center.

    Args:
        images (np.ndarray): Array of images with shape (n_images, 8, 8).
        angle (float): Rotation angle in degrees (positive = counterclockwise).

    Returns:
        np.ndarray: Rotated images, with the same shape as the original ones.
    """
    rotated = np.array([rotate(img, angle=angle, reshape=False, mode='nearest') for img in images])
    return np.clip(rotated, 0, 16)



def adjust_brightness(images, factor=1.2):
    """
    Adjusts the brightness of the images.

    Args:
        images (np.ndarray): Array of shape (n_images, 8, 8).
        factor (float): Brightness factor (>1 increases brightness, <1 decreases it).

    Returns:
        np.ndarray: Images with adjusted brightness.
    """
    bright_images = images * factor
    return np.clip(bright_images, 0, 16)



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