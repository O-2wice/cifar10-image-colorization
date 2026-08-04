"""Recover figures and metrics from an executed notebook.

Training happens on a Colab runtime whose filesystem is discarded when the
session ends. Moving the checkpoints off that runtime needs a browser attached
to the kernel, which the VS Code Colab extension does not provide.

The executed notebook is the way out. It travels back to the local machine on
save, carrying every figure as an embedded PNG and every metric in the printed
training logs. This script unpacks both, so the Quarto report can be rendered
from a notebook run without the model weights ever leaving Colab.

Usage:
    python scripts/extract_notebook_artifacts.py
    python scripts/extract_notebook_artifacts.py --notebook path/to.ipynb --check
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "original-image-colorization.ipynb"

# Figures are matched to filenames by the cell that produced them, so a rerun
# overwrites the same files instead of accumulating numbered duplicates.
FIGURE_NAMES = {
    "visualize_batch": "training-pairs",
    "plot_loss_curves": "loss-curves",
    "plot_predictions": "predictions",
    "visualize_feature_maps": "feature-maps",
    "visualize_fcn_weights": "fcn-weights",
}

EPOCH_PATTERN = re.compile(
    r"Epoch (\d+)/\d+ complete \| train ([\d.]+) \| val ([\d.]+)"
)
TEST_PATTERN = re.compile(r"Test Loss for (\w+) model: ([\d.]+)")


def cell_text(cell: dict) -> str:
    """Join a cell's stream output, which nbformat may store as a list."""
    parts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = output["text"]
            parts.append(text if isinstance(text, str) else "".join(text))
    return "".join(parts)


def source_of(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def figure_stem(cell: dict, fallback: int) -> str:
    """Name a figure after the plotting call in its cell."""
    source = source_of(cell)
    for marker, name in FIGURE_NAMES.items():
        if marker in source:
            return name
    return f"figure-{fallback}"


def extract_figures(notebook: dict, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Collect first so the totals per stem are known: a stem that produced one
    # figure gets a bare name, and one that produced several is numbered
    # throughout. Mixing the two conventions makes the set look truncated.
    collected: list[tuple[str, str]] = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        stem = figure_stem(cell, index)
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                collected.append((stem, encoded))

    totals: dict[str, int] = {}
    for stem, _ in collected:
        totals[stem] = totals.get(stem, 0) + 1

    written = []
    seen: dict[str, int] = {}
    for stem, encoded in collected:
        seen[stem] = seen.get(stem, 0) + 1
        suffix = f"-{seen[stem]}" if totals[stem] > 1 else ""
        path = figure_dir / f"{stem}{suffix}.png"
        path.write_bytes(base64.b64decode(encoded))
        written.append(path)

    return written


def extract_metrics(notebook: dict) -> tuple[dict, dict]:
    """Rebuild the loss history and test results from the printed logs."""
    histories: list[dict] = []
    test_results: dict[str, float] = {}

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        text = cell_text(cell)

        epochs = EPOCH_PATTERN.findall(text)
        if epochs:
            histories.append({
                "train_loss": [float(train) for _, train, _ in epochs],
                "val_loss": [float(val) for _, _, val in epochs],
            })

        for name, value in TEST_PATTERN.findall(text):
            test_results[f"{name.lower()}_test_loss"] = float(value)

    if len(histories) < 2:
        raise SystemExit(
            f"Expected two training histories in the notebook, found {len(histories)}. "
            "Run every training cell and save the notebook before extracting."
        )

    # The CNN trains first, so the histories arrive in that order.
    history = {"cnn": histories[0], "fcn": histories[1]}
    return history, test_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would be extracted without writing anything.",
    )
    args = parser.parse_args()

    if not args.notebook.exists():
        raise SystemExit(f"Notebook not found: {args.notebook}")

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    history, test_results = extract_metrics(notebook)

    executed = sum(
        1 for c in notebook["cells"]
        if c.get("cell_type") == "code" and c.get("outputs")
    )
    figures = sum(
        1 for c in notebook["cells"]
        for o in c.get("outputs", [])
        if "image/png" in o.get("data", {})
    )

    print(f"Notebook:    {args.notebook.name}")
    print(f"Cells with output: {executed}")
    print(f"Figures:     {figures}")
    for name, series in history.items():
        print(
            f"  {name}: {len(series['val_loss'])} epochs, "
            f"best validation {min(series['val_loss']):.4f}"
        )
    print(f"  test: {test_results}")

    if args.check:
        print("\nCheck only, nothing written.")
        return 0

    metric_dir = args.output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (metric_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8"
    )

    written = extract_figures(notebook, args.output_dir / "figures")

    print(f"\nWrote {len(written)} figures to {args.output_dir / 'figures'}")
    print(f"Wrote metrics to {metric_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
