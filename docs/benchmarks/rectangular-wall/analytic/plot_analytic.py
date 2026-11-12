"""Render current fields and convergence from the saved A0 verification data."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullFormatter
import numpy as np


BOX = (1.0, 2.2, -0.5, 0.5)
GRID = (24, 28)


def sections(root):
    archived = np.load(root / "archived-28x24.npz")
    repaired = np.load(root / "repaired-28x24.npz")
    exact = repaired["solovev_jphi_exact"].reshape(GRID) / 1e3
    previous = archived["solovev_jphi"].reshape(GRID) / 1e3
    current = repaired["solovev_jphi"].reshape(GRID) / 1e3
    previous_error, current_error = previous-exact, current-exact
    total_lo, total_hi = min(x.min() for x in (previous, current, exact)), max(x.max() for x in (previous, current, exact))
    abs_error = max(abs(previous_error).max(), abs(current_error).max())
    abs_corrected = abs(current_error).max()
    shared = TwoSlopeNorm(vmin=-abs_error, vcenter=0, vmax=abs_error)
    zoom = TwoSlopeNorm(vmin=-abs_corrected, vcenter=0, vmax=abs_corrected)
    fig, ax = plt.subplots(2, 3, figsize=(12.4, 7.0), layout="constrained")
    for a, value, title in zip(ax[0], (exact, previous, current), ("Analytic current", "Archived A0 current", "Repaired A0 current"), strict=True):
        total_plot = a.imshow(value, extent=BOX, origin="lower", cmap="viridis", vmin=total_lo, vmax=total_hi)
        a.set_title(title)
    for a, value, title, norm in zip(ax[1], (previous_error, current_error, current_error), ("Archived error", "Repaired error, shared scale", "Repaired error, expanded scale"), (shared, shared, zoom), strict=True):
        err_plot = a.imshow(value, extent=BOX, origin="lower", cmap="RdBu_r", norm=norm)
        a.set_title(title)
        if a is ax[1, 1]:
            shared_plot = err_plot
    fig.colorbar(total_plot, ax=ax[0], label=r"$J_\phi$ [kA/m²]")
    fig.colorbar(shared_plot, ax=ax[1, :2], label=r"$\Delta J_\phi$ [kA/m²]")
    fig.colorbar(err_plot, ax=ax[1, 2], label=r"$\Delta J_\phi$ [kA/m²]")
    for a in ax.flat:
        a.set_xlabel("R [m]")
        a.set_ylabel("Z [m]")
    fig.suptitle("Solov'ev field sampled into the rectangular wall complex · 28 × 24")
    fig.savefig(root / "solovev-current-sections.png", dpi=160)
    plt.close(fig)


def convergence(root):
    data = {tag: json.loads((root / f"{tag}.json").read_text())["grids"] for tag in ("archived", "repaired")}
    fig, ax = plt.subplots(figsize=(7.6, 4.7), layout="constrained")
    colors = ("#173d7a", "#d47617", "#764e88", "#24886c")
    regions = (("first_excluding_corners", "First layer excluding corners"), ("second", "Second layer"), ("corners", "Corners"), ("core", "Core"))
    x = np.array([7, 14, 28])
    for (key, label), color in zip(regions, colors, strict=True):
        for tag, style in (("archived", "o--"), ("repaired", "s-")):
            y = [v["cases"]["mixed_vacuum"]["Jphi_error_A_per_m2"][key]["rms"]/1e3 for v in data[tag].values()]
            ax.loglog(x, y, style, color=color, label=f"{label} · {tag}")
    ax.set_xticks(x, ["7 × 6", "14 × 12", "28 × 24"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r"$J_\phi$ error RMS [kA/m²]")
    ax.set_title("Analytic mixed vacuum field in a fixed physical box")
    ax.grid(which="both", alpha=.23)
    ax.legend(fontsize=8, ncols=2, loc="lower left")
    fig.savefig(root / "mixed-vacuum-current-convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sections(args.output)
    convergence(args.output)
