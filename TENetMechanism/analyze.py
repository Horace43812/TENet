"""Locate real class-pair degradation inside a frozen TENet layer."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import build_data, manifest, load_checkpoint, save_json, seed_all
from model import TENet


STAGES = ["input", "vertices", "nodes", "graph_raw", "graph_normalized", "projected",
          "pre_activation", "output", "content_input", "trace", "content_output"]


class Descriptor:
    """Shared spatial grid + fixed signed hash projection to equal probe dimensions.

    This is a measurement bottleneck, not an information-preservation guarantee.
    Identity-sensitive comparisons must be repeated with different grids/dimensions/seeds.
    """
    def __init__(self, grid, dim, seed):
        self.grid, self.dim, self.seed = grid, dim, seed
        self.maps = {}

    def __call__(self, value, detach=True):
        flat = F.adaptive_avg_pool2d(value.detach().float() if detach else value.float(), self.grid).flatten(1)
        size = flat.shape[1]
        key = (size, str(flat.device))
        if key not in self.maps:
            generator = torch.Generator().manual_seed(self.seed)
            index = torch.randint(self.dim, (size,), generator=generator).to(flat.device)
            sign = (torch.randint(2, (size,), generator=generator) * 2 - 1).float().to(flat.device)
            self.maps[key] = index, sign
        index, sign = self.maps[key]
        result = flat.new_zeros(len(flat), self.dim)
        result.scatter_add_(1, index.expand(len(flat), -1), flat * sign)
        return result.cpu()


@torch.no_grad()
def extract(model, dataset, args, device):
    layer = model.layers[args.layer]
    descriptor = Descriptor(args.grid, args.probe_dim, args.probe_seed)
    features = {name: [] for name in args.stages}
    labels, predictions = [], []
    def observe(name, value):
        if name in features:
            features[name].append(descriptor(value))
    layer.observer = observe
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model.eval()
    try:
        for images, target in loader:
            logits = model(images.to(device))
            labels.append(target)
            predictions.append(logits.argmax(1).cpu())
    finally:
        layer.observer = None
    return {key: torch.cat(value) for key, value in features.items()}, torch.cat(labels), torch.cat(predictions)


def fit_probe(train, labels, val, classes, args, device):
    seed_all(args.probe_seed)
    # Statistics are fitted only on train; val never selects an epoch/hyperparameter.
    mean = train.mean(0)
    std = train.std(0, unbiased=False).clamp_min(1e-5)
    train, val = (train - mean) / std, (val - mean) / std
    probe = nn.Linear(args.probe_dim, classes) if args.probe_hidden == 0 else nn.Sequential(
        nn.Linear(args.probe_dim, args.probe_hidden), nn.ReLU(), nn.Linear(args.probe_hidden, classes))
    probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_decay)
    for _ in range(args.probe_epochs):
        order = torch.randperm(len(labels))
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            loss = F.cross_entropy(probe(train[ids].to(device)), labels[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        probabilities = torch.cat([probe(batch.to(device)).softmax(1).cpu()
                                   for batch in val.split(args.batch_size)])
    state = {"mean": mean, "std": std, "state_dict": {k: v.cpu() for k, v in probe.state_dict().items()},
             "descriptor": {"grid": args.grid, "dim": args.probe_dim, "seed": args.probe_seed},
             "hidden": args.probe_hidden}
    return probabilities, state


def confusion(labels, prediction, classes):
    return torch.bincount(labels * classes + prediction, minlength=classes ** 2).reshape(classes, classes)


def split_validation(labels, seed):
    generator = torch.Generator().manual_seed(seed)
    discovery, confirmation = [], []
    for label in labels.unique(sorted=True):
        ids = (labels == label).nonzero(as_tuple=True)[0]
        if len(ids) < 2:
            raise ValueError("Analysis requires at least two validation images per class for discovery/confirmation")
        ids = ids[torch.randperm(len(ids), generator=generator)]
        cut = len(ids) // 2
        discovery.extend(ids[:cut].tolist())
        confirmation.extend(ids[cut:].tolist())
    return torch.tensor(sorted(discovery)), torch.tensor(sorted(confirmation))


def pair_changes(labels, before, after, ids, classes):
    a = confusion(labels[ids], before[ids], classes)
    b = confusion(labels[ids], after[ids], classes)
    pairs = {}
    for i in range(classes):
        for j in range(i + 1, classes):
            support = int(a[i].sum() + a[j].sum())
            old, new = int(a[i, j] + a[j, i]), int(b[i, j] + b[j, i])
            pairs[(i, j)] = {"support": support, "before_errors": old, "after_errors": new,
                             "delta_pp": (new - old) * 100 / support if support else 0}
    return pairs


def write_pairs(output, labels, predictions, discovery, confirmation, names, before, after, paths):
    old, new = predictions[before], predictions[after]
    first = pair_changes(labels, old, new, discovery, len(names))
    second = pair_changes(labels, old, new, confirmation, len(names))
    pairs = sorted(first, key=lambda key: (-first[key]["delta_pp"], key))
    with (output / "class_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_a", "class_b", "discovery_delta_pp", "confirmation_delta_pp",
                         "discovery_support", "discovery_before_errors", "discovery_after_errors",
                         "confirmation_support", "confirmation_before_errors", "confirmation_after_errors"])
        for i, j in pairs:
            a, b = first[(i, j)], second[(i, j)]
            writer.writerow([names[i], names[j], a["delta_pp"], b["delta_pp"], a["support"],
                a["before_errors"], a["after_errors"], b["support"], b["before_errors"], b["after_errors"])
    discovery_set = set(discovery.tolist())
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "partition", "path", "label", "before_prediction", "after_prediction", "transition"])
        for index, label in enumerate(labels.tolist()):
            a, b = int(old[index]), int(new[index])
            transition = "lost" if a == label and b != label else "recovered" if a != label and b == label else "other"
            writer.writerow([index, "discovery" if index in discovery_set else "confirmation", paths[index]["path"],
                             names[label], names[a], names[b], transition])
    return [{"classes": [names[i], names[j]], "discovery": first[(i, j)], "confirmation": second[(i, j)]}
            for i, j in pairs[:10]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", help="optional dataset relocation; relative paths/labels must match saved manifest")
    p.add_argument("--layer", type=int, default=0, help="zero-based graph layer index")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--before", choices=STAGES, default="input")
    p.add_argument("--after", choices=STAGES, default="output")
    for key, default in [("grid", 4), ("probe-dim", 256), ("probe-hidden", 0), ("probe-epochs", 30),
                         ("probe-seed", 123), ("batch-size", 32), ("workers", 0)]:
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-decay", type=float, default=1e-4)
    p.add_argument("--intervention", choices=["none", "no_trace", "no_cross", "no_graph_norm"], default="none")
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--output-dir")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.checkpoint = str(Path(args.checkpoint).resolve())
    if min(args.grid, args.probe_dim, args.probe_epochs, args.batch_size) < 1 or min(args.workers, args.probe_hidden) < 0:
        p.error("invalid probe sizes")
    if args.probe_lr <= 0 or args.probe_decay < 0:
        p.error("invalid probe optimization parameters")
    if args.before not in args.stages or args.after not in args.stages:
        p.error("before and after must be included in stages")
    if args.before == args.after or len(set(args.stages)) != len(args.stages):
        p.error("choose distinct before/after stages and do not repeat stages")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    model = TENet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if not 0 <= args.layer < len(model.layers):
        p.error("layer index outside model")
    if args.grid > checkpoint["model_config"]["image_size"] // (2 ** (args.layer + 1)):
        p.error("descriptor grid must not exceed the selected layer output spatial size")
    layer = model.layers[args.layer]
    if args.intervention == "no_trace" and not layer.use_trace_branch:
        p.error("trace branch already disabled")
    if args.intervention == "no_cross" and not layer.use_cross_stream_guidance:
        p.error("cross guidance already disabled")
    if args.intervention == "no_graph_norm" and layer.use_chebyshev:
        p.error("no_graph_norm is only defined for the single-Laplacian mode")
    data_args = SimpleNamespace(**checkpoint["args"])
    data_args.batch_size, data_args.workers = args.batch_size, args.workers
    if args.data_dir:
        data_args.data_dir = args.data_dir
    seed_all(data_args.seed)
    data = build_data(data_args, augment=False)
    samples = manifest(data)
    saved = checkpoint["manifest"]
    def relative_manifest(value, root):
        return {"class_names": value["class_names"], **{split: [
            {"path": str(Path(item["path"]).relative_to(Path(root).resolve())).replace("\\", "/")
             if not item["path"].startswith("fake:") else item["path"], "label": item["label"]}
            for item in value[split]] for split in ("train", "val")}}
    if samples != saved:
        if not args.data_dir or relative_manifest(samples, args.data_dir) != relative_manifest(saved, checkpoint["args"]["data_dir"]):
            raise ValueError("Dataset files/classes differ from the checkpoint manifest")
    output = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "analysis" / (
        f"layer{args.layer}_{args.intervention}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", {"analysis": vars(args), "training": checkpoint["args"],
                                      "model": checkpoint["model_config"], "checkpoint_epoch": checkpoint["epoch"]})
    # Baseline extraction and fits always use the unmodified frozen checkpoint.
    print("Extracting deterministic train/val descriptors...", flush=True)
    train, train_labels, _ = extract(model, data.train_loader.dataset, args, device)
    val, labels, model_prediction = extract(model, data.eval_loader.dataset, args, device)
    discovery, confirmation = split_validation(labels, args.probe_seed)
    probabilities, probes, predictions, metrics = {}, {}, {}, {}
    for name in args.stages:
        print(f"Fitting probe: {name}", flush=True)
        probabilities[name], probes[name] = fit_probe(train[name], train_labels, val[name], data.num_classes, args, device)
        predictions[name] = probabilities[name].argmax(1)
        metrics[name] = {"accuracy": float((predictions[name] == labels).float().mean() * 100),
                         "descriptor_dim": args.probe_dim}
    top_pairs = write_pairs(output, labels, predictions, discovery, confirmation, data.class_names,
                            args.before, args.after, samples["val"])
    intervention_report = None
    if args.intervention != "none":
        layer.intervention = args.intervention
        changed, changed_labels, changed_model = extract(model, data.eval_loader.dataset, args, device)
        if not torch.equal(labels, changed_labels):
            raise RuntimeError("Intervention sample order changed")
        changed_predictions = {}
        for name in args.stages:
            state = probes[name]
            probe = nn.Linear(args.probe_dim, data.num_classes) if args.probe_hidden == 0 else nn.Sequential(
                nn.Linear(args.probe_dim, args.probe_hidden), nn.ReLU(), nn.Linear(args.probe_hidden, data.num_classes))
            probe.load_state_dict(state["state_dict"])
            probe.eval()
            with torch.no_grad():
                changed_predictions[name] = torch.cat([probe(batch).argmax(1) for batch in
                    ((changed[name] - state["mean"]) / state["std"]).split(args.batch_size)])
        intervention_report = {"name": args.intervention,
            "baseline_model_accuracy": float((model_prediction == labels).float().mean() * 100),
            "intervened_model_accuracy": float((changed_model == labels).float().mean() * 100),
            "fixed_probe_accuracy": {name: float((value == labels).float().mean() * 100)
                                      for name, value in changed_predictions.items()}}
        torch.save({"labels": labels, "model_baseline": model_prediction, "model_intervened": changed_model,
                    "probe_predictions": changed_predictions}, output / "intervention.pth")
        layer.intervention = None
    torch.save({"labels": labels, "probabilities": probabilities, "probes": probes,
                "discovery": discovery, "confirmation": confirmation, "model_predictions": model_prediction},
               output / "probes.pth")
    if args.save_features:
        torch.save({"train": train, "train_labels": train_labels, "val": val, "val_labels": labels}, output / "features.pth")
    save_json(output / "confusion.json", {name: confusion(labels, pred, data.num_classes).tolist()
                                         for name, pred in predictions.items()})
    save_json(output / "report.json", {"metrics": metrics, "top_pairs": top_pairs, "intervention": intervention_report,
        "class_names": data.class_names, "discovery_ids": discovery.tolist(), "confirmation_ids": confirmation.tolist()})
    lines = ["# TENet mechanism analysis", "", f"Layer {args.layer}: {args.before} -> {args.after}", "",
             "| Stage | Frozen-feature probe accuracy (%) |", "| --- | ---: |"]
    lines += [f"| {name} | {value['accuracy']:.2f} |" for name, value in metrics.items()]
    lines += ["", "## Class pairs ranked ONLY on discovery subset", "",
              "| Class pair | Discovery confusion increase (pp) | Confirmation increase (pp) |", "| --- | ---: | ---: |"]
    lines += [f"| {' / '.join(row['classes'])} | {row['discovery']['delta_pp']:.2f} | {row['confirmation']['delta_pp']:.2f} |"
              for row in top_pairs]
    lines += ["", "Positive delta = increased mutual confusion; other errors are not included in this pair metric.",
              "See samples.csv for lost/recovered predictions and original image paths.",
              "This measures probe accessibility, not information loss or causal importance.",
              "Descriptor pooling/projection can itself hide distinctions; repeat with multiple grids, dimensions and seeds.",
              "Validation may already have selected the network checkpoint. Confirmation is exploratory, not an untouched test set.",
              "Interventions freeze the network AND probes; changes may reflect distribution shift, not a repaired mechanism."]
    if intervention_report:
        lines += ["", "## Frozen intervention", "", "```json", json.dumps(intervention_report, indent=2), "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis saved: {output.resolve()}")


if __name__ == "__main__":
    main()
