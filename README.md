# CIFAR-10 Image Colorization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/cifar10-image-colorization/blob/main/notebooks/original-image-colorization.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

This project trains neural networks to reconstruct RGB CIFAR-10 images from grayscale inputs. It compares:

- a shallow convolutional neural network that preserves spatial image structure
- a fully connected baseline that maps flattened grayscale pixels directly to RGB pixels

The main runnable notebook is:

```text
notebooks/original-image-colorization.ipynb
```

The Quarto report in `index.qmd` is the presentation layer for GitHub Pages.

## Project Structure

```text
cifar10-image-colorization/
  index.qmd                         # Quarto project report
  _quarto.yml                       # Quarto site configuration
  requirements.txt                  # Python dependencies
  COLAB.md                          # Colab workflow from VS Code
  notebooks/
    original-image-colorization.ipynb # Main runnable project notebook
  scripts/
    train_colorization.py           # Terminal training script with progress logs
  outputs/
    models/                         # Local trained checkpoints, ignored by git
    metrics/                        # JSON metrics and training history
    figures/                        # Optional exported figures
```

## Recommended Workflow

Use the notebook when you want to relearn the project and train on Colab GPU:

```text
notebooks/original-image-colorization.ipynb
```

Inside the notebook, run cells from top to bottom. The final Colab cell packages `outputs/` as a zip so the trained checkpoints and metrics can be copied back into this repository.

The training cells are built to survive a dropped Colab session. Each epoch writes a resume checkpoint holding model weights, optimizer state and loss history, so re-running an interrupted training cell continues from the epoch it reached instead of starting over. The two models train in separate cells, so a failure on one cannot cost you the other.

Once `outputs/` holds real artifacts, render the report:

```powershell
quarto render
```

The rendered HTML site is written to `docs/`, and the executed results are cached under `_freeze/`. Both are committed, so the published page keeps its figures even when cloned onto a machine with no GPU, no CIFAR-10 download and no checkpoints. Add `--force` when you deliberately want to re-execute.

## Methods

The task is framed as pixel-level reconstruction:

- input: grayscale CIFAR-10 image, shape `1 x 32 x 32`
- target: RGB CIFAR-10 image, shape `3 x 32 x 32`
- loss: mean squared error

The CIFAR-10 training split is divided into 45,000 training samples and 5,000 validation samples. The 10,000-image CIFAR-10 test split is held out for final evaluation.

The CNN uses four convolutional layers with ReLU activations and a sigmoid output. The fully connected baseline uses a linear projection from `32 x 32` grayscale pixels to `3 x 32 x 32` RGB pixels.

Both models end in a sigmoid, so the RGB targets are kept in `[0, 1]` with `ToTensor()` and **no** `Normalize()`. Normalizing targets to `[-1, 1]` would place half the target range outside anything a sigmoid can output and floor the achievable MSE near `0.15` regardless of architecture. The Quarto report works through this in the preprocessing section.

## Portfolio Notes

This repository is designed to show:

- PyTorch dataset construction
- image-to-image training
- model comparison
- training curve interpretation
- visual evaluation
- feature-map and weight inspection

Use real CIFAR-10 only. Data and heavyweight trained artifacts stay local unless intentionally added later.
