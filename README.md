# Opti4ML-TidyTeam

This project explores the impact of data ordering and sampling strategies on model training performance, with a focus on curriculum learning and other advanced sampling techniques.

## Notebooks

The repository contains three main Jupyter notebooks, each corresponding to a specific dataset:

- `toy_dataset.ipynb`: A synthetic dataset used for initial exploration and visualization.
- `mnist_dataset.ipynb`: Experiments conducted on the MNIST handwritten digits dataset.
- `cifar_10_dataset.ipynb`: Experiments conducted on the CIFAR-10 image classification dataset.

## Objectives

The primary goal is to investigate how different data sampling orders affect model training, particularly in the presence of noise. For each dataset, we analyze and compare several data sampling strategies.

## Sampling Strategies Analyzed

Each notebook is structured to evaluate the following techniques:

- **Base Case**: Random sampling, used as a control baseline.
- **Curriculum Learning**: Training begins with "easy" examples and gradually includes harder ones.
  - *Cumulative*: Previously seen data is retained across stages.
  - *Strict*: Each stage includes only a specific difficulty range.
- **Reverse Curriculum Learning**: The opposite of curriculum learning—training starts with harder examples.
  - *Cumulative and Strict* versions are both explored.
- **Hard Example Mining**: Prioritizing the hardest examples during training.
- **Stratified Monte Carlo Sampling**: Sampling from different difficulty strata using Monte Carlo techniques.

## Noise Robustness

To test robustness on noisy data, we introduced two types of noise to both MNIST and CIFAR-10:

- **Gaussian Noise**
- **Impulse (Salt-and-Pepper) Noise**

Noisy variants of the datasets were used to have a precise difficulty assessment for each data point (with different noise levels).
