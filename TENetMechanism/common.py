import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import build_dataloaders


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_checkpoint(path, device):
    # Only load checkpoints produced by this workspace that you trust.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=device)


def build_data(args, augment=None):
    return build_dataloaders(args.dataset, args.data_dir, args.image_size, args.batch_size, args.workers,
        not args.no_aug if augment is None else augment, args.num_classes,
        args.fake_train_size, args.fake_eval_size, args.class_start, args.class_count,
        args.train_per_class, args.val_per_class, args.seed, args.common_classes_only)


def manifest(data):
    def items(dataset):
        if hasattr(dataset, "samples"):
            return [{"path": str(Path(path).resolve()), "label": label} for path, label in dataset.samples]
        return [{"path": f"fake:{i}", "label": int(dataset[i][1])} for i in range(len(dataset))]
    return {"class_names": data.class_names, "train": items(data.train_loader.dataset),
            "val": items(data.eval_loader.dataset)}


def evaluate(model, loader, device, optimizer=None, smoothing=0.0):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {"loss": loss_sum / count, "accuracy": correct * 100 / count}
