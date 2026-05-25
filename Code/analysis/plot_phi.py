"""Render phi heatmap PNGs from saved phi_*.pt files.

Usage:
    python -m analysis.plot_phi <run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir: str) -> None:
    d = Path(run_dir)
    files = sorted(d.glob("phi_*.pt"))
    if not files:
        print(f"[plot_phi] no phi_*.pt under {d}")
        return
    for f in files:
        grid = torch.load(f, map_location="cpu")
        arr = grid.numpy()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(arr, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", aspect="auto")
        ax.set_xlabel("v_j")
        ax.set_ylabel("v_i")
        ax.set_title(f"{f.stem} (ch0)")
        fig.colorbar(im, ax=ax)
        out = f.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot_phi] wrote {out}  min={arr.min():.3f} max={arr.max():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
