"""Generate the social preview card and favicon for the report site.

Without an `og:image`, a link to the report unfurls in Slack, LinkedIn and X as a bare
grey box. This renders the 1200x630 card those platforms expect, plus a favicon.

Numbers are read from the metrics the notebook wrote, so the card cannot drift away
from the reported results.

    python scripts/make_social_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics" / "test_results.json"
OUT = ROOT / "assets"

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"

CNN_PARAMS = 150_019
FCN_PARAMS = 3_148_800


def social_card() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    cnn, fcn = m["cnn_test_loss"], m["fcn_test_loss"]

    mpl.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.06, 0.85, " ".join("CIFAR-10\u2009IMAGE\u2009COLORIZATION"),
             fontsize=13, color=BLUE, fontweight="700")
    ratio = round(FCN_PARAMS / CNN_PARAMS)
    fig.text(0.06, 0.705, f"{ratio}\u00d7 the parameters.",
             fontsize=45, color=INK, fontweight="700")
    fig.text(0.06, 0.585, "Worse pictures.", fontsize=45, color=INK, fontweight="700")
    fig.text(0.06, 0.485,
             "A convolutional colorizer against a fully connected baseline, "
             "same task and data.",
             fontsize=16.5, color=MUTED)

    tiles = [
        ("CNN test MSE", f"{cnn:.4f}", f"{CNN_PARAMS:,} params", GREEN),
        ("Fully connected", f"{fcn:.4f}", f"{FCN_PARAMS:,} params", ORANGE),
        ("Error reduction", f"{(1 - cnn / fcn) * 100:.0f}%", f"with {ratio}x fewer weights", BLUE),
    ]
    for i, (label, value, sub, colour) in enumerate(tiles):
        x = 0.06 + i * 0.305
        fig.patches.append(
            mpl.patches.Rectangle((x, 0.11), 0.006, 0.20, transform=fig.transFigure,
                                  facecolor=colour, edgecolor="none")
        )
        fig.text(x + 0.022, 0.265, label, fontsize=13.5, color=MUTED)
        fig.text(x + 0.022, 0.165, value, fontsize=34, color=INK, fontweight="700")
        fig.text(x + 0.022, 0.122, sub, fontsize=12, color=colour, fontweight="600")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "social-card.png", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    print("wrote assets/social-card.png (1200x630)")


def favicon() -> None:
    """Grey square turning colour: the task in one mark."""
    fig = plt.figure(figsize=(0.64, 0.64), dpi=100)
    fig.patch.set_facecolor("#8a8a8a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpl.patches.Rectangle((0.5, 0), 0.5, 1, facecolor=BLUE))
    ax.add_patch(mpl.patches.Rectangle((0.18, 0.18), 0.30, 0.30, facecolor="#d4d4d4"))
    ax.add_patch(mpl.patches.Rectangle((0.52, 0.52), 0.30, 0.30, facecolor="#eda100"))
    fig.savefig(OUT / "favicon.png", facecolor="#8a8a8a", dpi=100)
    plt.close(fig)
    print("wrote assets/favicon.png (64x64)")


if __name__ == "__main__":
    social_card()
    favicon()
