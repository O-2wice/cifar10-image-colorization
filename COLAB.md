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

## Workflow (VS Code)

1. Open this repository in VS Code.
2. Open `notebooks/original-image-colorization.ipynb`.
3. In the notebook toolbar, choose `Select Kernel > Colab > Auto Connect`.
4. Sign in when prompted.
5. Pick a GPU runtime.
6. Run the cells from top to bottom. The setup cell prints `Runtime accelerator: GPU` and `Running in Colab: True` when the connection is live. Check both before starting a long run.
7. Run the Drive mount cell before training. This is how results get off the runtime on this route.
8. The training cells write best checkpoints to `outputs/models/` and metrics to `outputs/metrics/` **on the Colab machine**, not on your laptop.
9. Run the final cell. With Drive mounted it copies `image-colorization-outputs.zip` to `MyDrive/cifar10-image-colorization/`.
10. Download that zip from Drive and extract it over the local `outputs/` folder.
11. Render the report locally:

```powershell
quarto render
```

## Browser Colab vs the VS Code Extension

Both attach to the same kind of Colab VM, but they differ in one way that matters here: the browser has a bridge that can push a file straight to your downloads folder, and the VS Code extension does not.

The notebook handles both. It detects the runtime by checking whether the `google.colab` package is *installed*, not whether it has been imported. The browser frontend imports it at startup; the VS Code extension attaches a plain kernel to the same machine and does not. A `sys.modules` check would report "not Colab" on a real Colab GPU and silently disable the DataLoader workers.

The final cell tries Drive first, then a direct browser download, and otherwise reports where the archive sits on the runtime.

## Your Files Are Not On The Runtime

The notebook is edited locally but executes remotely, so the Colab machine does not see your repository. This is fine: the notebook downloads CIFAR-10 itself and writes everything relative to the runtime's working directory. It does mean anything you want to keep has to be copied off deliberately, which is what the Drive step is for.

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
