"""Plot pixel sensitivity of a frozen class-pair probe before/after aggregation."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from analyze import Descriptor
from common import load_checkpoint, save_json
from model import build_model


def make_probe(state, classes):
    dim, hidden = state["descriptor"]["dim"], state["hidden"]
    result = nn.Linear(dim, classes) if hidden == 0 else nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, classes))
    result.load_state_dict(state["state_dict"])
    return result.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True, type=Path)
    p.add_argument("--class-a", required=True, help="exact class folder name")
    p.add_argument("--class-b", required=True)
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--partition", choices=["discovery", "confirmation"], default="discovery")
    args = p.parse_args()
    if args.max_samples < 1:
        p.error("max-samples must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((args.analysis_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((args.analysis_dir / "report.json").read_text(encoding="utf-8"))
    names = report["class_names"]
    if args.class_a == args.class_b or any(name not in names for name in (args.class_a, args.class_b)):
        p.error("provide two different class names from report.json")
    a, b = names.index(args.class_a), names.index(args.class_b)
    checkpoint = load_checkpoint(config["analysis"]["checkpoint"], "cpu")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = load_checkpoint(args.analysis_dir / "probes.pth", "cpu")
    stages = [config["analysis"]["before"], config["analysis"]["after"]]
    probes = {name: make_probe(saved["probes"][name], len(names)) for name in stages}
    if checkpoint["args"]["dataset"] != "imagefolder":
        p.error("image visualization requires an ImageFolder run")
    with (args.analysis_dir / "samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["label"] in (args.class_a, args.class_b)
                and row["partition"] == args.partition]
    rows.sort(key=lambda row: (row["transition"] != "lost", int(row["index"])))
    rows = rows[:args.max_samples]
    if not rows:
        p.error("no samples in this partition for the selected pair")
    output = args.analysis_dir / f"pair_{args.class_a}_{args.class_b}_{args.partition}"
    output.mkdir(exist_ok=False)
    size = checkpoint["model_config"]["image_size"]
    transform = transforms.Compose([transforms.Resize(round(size / 0.875)), transforms.CenterCrop(size), transforms.ToTensor()])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    records = []
    for row in rows:
        with Image.open(row["path"]) as source:
            rgb = transform(source.convert("RGB")).unsqueeze(0)
        pixels = rgb.clone().requires_grad_(True)
        features = {}
        def observe(name, value):
            if name in probes:
                setting = saved["probes"][name]["descriptor"]
                features[name] = Descriptor(setting["grid"], setting["dim"], setting["seed"])(value, detach=False)
        model.layers[config["analysis"]["layer"]].observer = observe
        model((pixels - mean) / std)
        maps, margins = [], {}
        for name in stages:
            state = saved["probes"][name]
            logits = probes[name]((features[name] - state["mean"]) / state["std"])
            margin = logits[0, a] - logits[0, b]
            grad = torch.autograd.grad(margin, pixels, retain_graph=True)[0].abs().mean(1)[0]
            maps.append(grad.detach())
            margins[name] = float(margin.detach())
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(rgb[0].permute(1, 2, 0))
        axes[0].set_title(f"{row['label']} / {row['transition']}")
        for index, name in enumerate(stages, 1):
            axes[index].imshow(maps[index - 1], cmap="magma")
            axes[index].set_title(f"{name}\nmargin A-B={margins[name]:.3f}")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output / f"sample_{row['index']}.png", dpi=160)
        plt.close(figure)
        torch.save({"pixel_gradients": dict(zip(stages, maps)), "margins": margins}, output / f"sample_{row['index']}.pth")
        records.append({**row, "margins": margins})
    save_json(output / "samples.json", records)
    (output / "README.txt").write_text(
        "Gradients measure local sensitivity of frozen class-pair margins to RGB pixels, not causal importance.\n"
        "Heatmaps use separate color scales; raw gradients are saved. Only the center crop is shown.\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
