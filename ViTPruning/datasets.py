"""Dataset factory for the ViT pruning platform."""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_classes: int
    train_samples: int
    eval_samples: int


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


def _imagefolder_datasets(root: Path, image_size: int, augment: bool) -> tuple[Dataset, Dataset, int]:
    train_root, eval_root = root / "train", root / "val"
    if not train_root.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError("ImageFolder needs '<data-dir>/train/<class>/' and '<data-dir>/val/<class>/'")
    train_set = datasets.ImageFolder(train_root, transform=_train_transform(image_size, augment))
    eval_set = datasets.ImageFolder(eval_root, transform=_eval_transform(image_size))
    if train_set.classes != eval_set.classes:
        raise ValueError("ImageFolder train and val splits must have identical class directories")
    return train_set, eval_set, len(train_set.classes)


def build_dataloaders(name, data_dir, image_size, batch_size, workers, augment, num_classes, fake_train_size, fake_eval_size):
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
        train_set, eval_set, num_classes = _imagefolder_datasets(Path(data_dir), image_size, augment)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: fake, cifar10, imagefolder")
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    return DataBundle(DataLoader(train_set, shuffle=True, **options), DataLoader(eval_set, shuffle=False, **options),
                      num_classes, len(train_set), len(eval_set))
