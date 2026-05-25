"""Run input-grad saliency over a few CIFAR-10 test samples for a saved TENet.

Usage:
    python -m analysis.run_input_grad <run_dir> --alpha 32 --num-layers 2 --num-samples 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis.hooks import iter_graphified_layers
from analysis.input_grad import input_grad_saliency
from model_rgb32 import TENet_RGB32


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        print(f"[err] missing {ckpt}")
        sys.exit(1)

    model = TENet_RGB32(
        num_classes=10,
        num_layers=args.num_layers,
        input_size=32,
        alpha=args.alpha,
        use_trace_branch=True,
        use_edge_enhance=True,
        use_cross_stream_guidance=True,
        use_directed_flow=False,
        use_chebyshev=False,
        use_spatial_bridge=False,
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    layer_names = [name for name, _ in iter_graphified_layers(model)]
    num_layers = len(layer_names)
    print(f"[model] layers: {layer_names}")

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tf = transforms.Compose([transforms.ToTensor(), norm])
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    loader = DataLoader(test, batch_size=1, shuffle=False)

    out_dir = run_dir / "input_grad"
    out_dir.mkdir(exist_ok=True)

    # Denormalisation for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    it = iter(loader)
    for s in range(args.num_samples):
        x, y = next(it)
        img = (x[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, num_layers + 1, figsize=(3 * (num_layers + 1), 3))
        axes[0].imshow(img)
        axes[0].set_title(f"input (y={int(y)})")
        axes[0].axis("off")

        for li in range(num_layers):
            sal = input_grad_saliency(model, x, layer_index=li, target_ij=None, stream="topo")
            heat = sal.abs().sum(dim=0).numpy()
            ax = axes[li + 1]
            ax.imshow(img, alpha=0.5)
            ax.imshow(heat, cmap="hot", alpha=0.5)
            ax.set_title(f"{layer_names[li]} |grad|@center")
            ax.axis("off")

        fig.tight_layout()
        out = out_dir / f"sample_{s:02d}_y{int(y)}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[sal] wrote {out}")


if __name__ == "__main__":
    main()
