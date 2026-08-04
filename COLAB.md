# Training on Colab from VS Code

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/cifar10-image-colorization/blob/main/notebooks/original-image-colorization.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

This repository uses one notebook for Colab training and project review:

```text
notebooks/original-image-colorization.ipynb
```

There is no separate Colab-only notebook. The badge opens the notebook directly once the repository exists on GitHub.

## Why Use Colab?

Local CPU training is slow for this project. Colab lets the expensive training run on hosted GPU hardware while the project files stay organized locally.

## Workflow

1. Open this repository in VS Code.
2. Open `notebooks/original-image-colorization.ipynb`.
3. In the notebook toolbar, choose `Select Kernel > Colab > Auto Connect`.
4. Sign in when prompted.
5. In Colab, set the runtime to GPU.
6. Run the notebook cells from top to bottom.
7. The training cells save best checkpoints under `outputs/models/` and metrics under `outputs/metrics/`.
8. Run the final notebook cell to download `image-colorization-outputs.zip`.
9. Extract that zip into the local project folder so `outputs/` is refreshed.
10. Render the report:

```powershell
quarto render
```

## If the Session Drops

Colab disconnects on idle browsers and caps session length, and everything under `/content` disappears with the runtime.

The notebook is built for this. Every epoch writes `last_model_*.pth` alongside the best checkpoint, carrying model weights, optimizer state and loss history. If the runtime dies partway through:

1. Reconnect and re-run the setup cells.
2. Re-run the training cell that was interrupted.
3. It reports `Resuming ... at epoch N` and continues from there.

The CNN and the baseline train in separate cells, so losing one does not cost the other. Download the outputs zip as soon as training finishes rather than leaving artifacts on the runtime.

## Notes

- The notebook uses real CIFAR-10 images.
- The setup cell prints whether the runtime is using CPU or GPU, and whether it is running in Colab.
- DataLoader workers are enabled only on Colab. The per-sample work (PIL decode, grayscale conversion, tensor casts) is CPU-bound and becomes the bottleneck on a GPU runtime; Linux forks workers cheaply, while Windows spawns them and cannot unpickle a Dataset class defined inside a notebook.
- Local CPU training measures roughly 9 minutes per epoch for the CNN, which is why the GPU path exists.
