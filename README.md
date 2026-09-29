# CIFAR-10 Image Colorization

[![Read the report](https://img.shields.io/badge/read-the%20report-2b6cb0)](https://o-2wice.github.io/cifar10-image-colorization/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/cifar10-image-colorization/blob/main/notebooks/original-image-colorization.ipynb)
![Runtime](https://img.shields.io/badge/runtime-CPU%20or%20GPU-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

Predict the colors of a grayscale image. Two PyTorch models are trained on the
same task, loss and data: a convolutional colorizer and a fully connected
baseline. The baseline has about 21x more parameters and still scores worse.

**[Read the write-up](https://o-2wice.github.io/cifar10-image-colorization/)** for
the method, the figures and the results.

|                   | CNN         | Fully connected baseline |
| ----------------- | ----------- | ------------------------ |
| Parameters        | 150,019     | 3,148,800                |
| Held-out test MSE | **0.0053**  | 0.0074                   |

## Quick Start

```powershell
pip install -r requirements.txt
```

Open `notebooks/original-image-colorization.ipynb` and run it top to bottom.
CPU training takes about nine minutes per epoch. For GPU, see [COLAB.md](COLAB.md),
which covers running on a Colab runtime from the browser or from VS Code.

Each epoch writes a resume checkpoint with model weights, optimizer state and
loss history. If a session drops, re-running the training cell continues from
the last completed epoch. The two models train in separate cells.

To run it from a terminal instead:

```powershell
python scripts/train_colorization.py --epochs 10
```

## Report

`index.qmd` is the written version of the experiment. Render it with:

```powershell
quarto render
```

HTML goes to `docs/` for GitHub Pages, and executed results are cached in
`_freeze/`. Both are committed so the page keeps its figures on a machine
without a GPU, the dataset or checkpoints. Use `--force` to re-execute.

Re-executing needs the kernel named in `index.qmd`. Register it once from this
repo's environment:

```powershell
python -m ipykernel install --user --name cifar10-image-colorization
```

Rendering from the cache does not need it, which is why a plain `quarto render`
works on a fresh clone.

## Layout

```text
index.qmd                             # write-up, renders to docs/
notebooks/
  original-image-colorization.ipynb   # the experiment
scripts/
  train_colorization.py               # terminal version
outputs/
  models/                             # checkpoints (gitignored)
  metrics/                            # loss history, test results
```

## Method

Pixel-level regression. Input is a `1 x 32 x 32` grayscale tensor, output is a
`3 x 32 x 32` RGB tensor, scored with mean squared error.

The CNN uses four `3 x 3` convolutions with ReLU and a sigmoid output. There is
no pooling, so the image stays at full resolution the whole way through. The
baseline flattens the input and applies one linear layer mapping 1,024 grayscale
values to 3,072 RGB values.

The 50,000 training images are split 45,000 / 5,000. Early stopping and
checkpoint selection use the validation split only. The 10,000 test images are
used once, at the end.

## Target Scaling

Both models end in a sigmoid, so their outputs fall in `[0, 1]`. Targets are
produced with `ToTensor()` and are not normalized.

The coursework version normalized targets to `[-1, 1]`. A sigmoid cannot reach
the negative half of that range, so the error floored near `0.15` regardless of
how long the models trained. Removing the normalization is what makes the loss
values here meaningful. The report shows the arithmetic.

## Notes

The dataset and trained weights are not committed. Everything needed to
regenerate them is.

## License

[MIT](LICENSE). The dataset and any pretrained weights keep their own licences.
