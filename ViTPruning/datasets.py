"""Dataset factory for the ViT pruning platform."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch


@dataclass
class DataBundle:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_classes: int
    train_samples: int
    eval_samples: int
    class_names: list[str]


def _normalization():
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def _train_transform(image_size: int, augment: bool):
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.extend([transforms.RandomCrop(image_size, padding=max(1, image_size // 8)), transforms.RandomHorizontalFlip()])
    steps.extend([transforms.ToTensor(), _normalization()])
    return transforms.Compose(steps)


def _eval_transform(image_size: int):
    steps = [transforms.Resize((image_size, image_size))]
    steps.extend([transforms.ToTensor(), _normalization()])
    return transforms.Compose(steps)


class SelectedImageFolder(datasets.ImageFolder):
    """Only scan selected class directories; keep contiguous, shared labels."""

    def __init__(self, root, classes, transform):
        self.selected_classes = classes
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        return self.selected_classes, {name: i for i, name in enumerate(self.selected_classes)}


def _limit_per_class(dataset, limit, seed):
    if not limit:
        return
    rng = random.Random(seed)
    groups = [[] for _ in dataset.classes]
    for sample in dataset.samples:
        groups[sample[1]].append(sample)
    dataset.samples = [sample for group in groups
                       for sample in sorted(rng.sample(group, min(limit, len(group))))]
    dataset.imgs = dataset.samples
    dataset.targets = [target for _, target in dataset.samples]


def _imagefolder_datasets(root: Path, image_size: int, augment: bool,
                          class_start=0, class_count=0, train_per_class=0,
                          val_per_class=0, seed=42):
    train_root, eval_root = root / "train", root / "val"
    if not train_root.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError("ImageFolder needs '<data-dir>/train/<class>/' and '<data-dir>/val/<class>/'")
    classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    val_classes = sorted(path.name for path in eval_root.iterdir() if path.is_dir())
    if classes != val_classes:
        raise ValueError("ImageFolder train and val splits must have identical class directories")
    classes = classes[class_start:class_start + class_count if class_count else None]
    if len(classes) < 2:
        raise ValueError("Select at least two classes; check --class-start and --class-count")
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    eval_transform = transforms.Compose([
        transforms.Resize(round(image_size / 0.875)), transforms.CenterCrop(image_size),
        transforms.ToTensor(), normalization])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalization]) if augment else eval_transform
    train_set = SelectedImageFolder(train_root, classes, train_transform)
    eval_set = SelectedImageFolder(eval_root, classes, eval_transform)
    _limit_per_class(train_set, train_per_class, seed)
    _limit_per_class(eval_set, val_per_class, seed + 1)
    return train_set, eval_set, len(classes)


def build_dataloaders(name, data_dir, image_size, batch_size, workers, augment, num_classes, fake_train_size, fake_eval_size,
                      class_start=0, class_count=0, train_per_class=0, val_per_class=0, seed=42):
    """Build a registered dataset. ``fake`` validates the pipeline without data."""
    if name == "fake":
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        train_set = datasets.FakeData(size=fake_train_size, image_size=(3, image_size, image_size),
                                      num_classes=num_classes, transform=_train_transform(image_size, augment), random_offset=0)
        eval_set = datasets.FakeData(size=fake_eval_size, image_size=(3, image_size, image_size),
                                     num_classes=num_classes, transform=_eval_transform(image_size), random_offset=fake_train_size)
    elif name == "cifar10":
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=_train_transform(image_size, augment))
        eval_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=_eval_transform(image_size))
        num_classes = 10
    elif name == "imagefolder":
        train_set, eval_set, num_classes = _imagefolder_datasets(
            Path(data_dir), image_size, augment, class_start, class_count, train_per_class, val_per_class, seed)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: fake, cifar10, imagefolder")
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                   worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(seed))
    return DataBundle(DataLoader(train_set, shuffle=True, **options), DataLoader(eval_set, shuffle=False, **options),
                      num_classes, len(train_set), len(eval_set),
                      list(getattr(train_set, "classes", [str(i) for i in range(num_classes)])))


def _seed_worker(worker_id):
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
