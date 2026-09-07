"""Immutable, content-verified train/dev/final manifests independent of training seeds."""
import hashlib
import json
import random
from pathlib import Path


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_manifest(root, class_count=50, class_start=0, train_limit=500, data_seed=42,
                     split_seed=20260906, final_unseen=False, groups_file=None):
    root = Path(root).resolve()
    from PIL import Image
    if min(class_count, class_start, train_limit) < 0:
        raise ValueError("Class/sample limits must be nonnegative")
    if not all((root / split).is_dir() for split in ("train", "val")):
        raise ValueError(f"Expected train/<class> and val/<class> in {root}")
    names = sorted({p.name for p in (root / "train").iterdir() if p.is_dir()} &
                   {p.name for p in (root / "val").iterdir() if p.is_dir()})
    names = names[class_start:class_start + class_count if class_count else None]
    if len(names) < 2:
        raise ValueError("Need at least two shared classes")
    # Optional mapping of relative image paths to source/subject groups.
    groups = read_json(groups_file) if groups_file else {}
    records, seen = {"train": [], "val": []}, {}
    for split in records:
        for label, name in enumerate(names):
            for path in sorted((root / split / name).rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
                    continue
                with Image.open(path) as picture:
                    picture.verify()
                sha = digest_file(path)
                relative = path.relative_to(root).as_posix()
                if sha in seen:
                    raise ValueError(f"Duplicate image content: {seen[sha]} and {relative}. Deduplicate before init.")
                seen[sha] = relative
                records[split].append({"path": relative, "label": label, "sha256": sha,
                                       "group": str(groups.get(relative, relative))})
    final_groups = {r["group"] for r in records["val"]}
    if final_groups & {r["group"] for r in records["train"]}:
        raise ValueError("Source groups overlap train and original val")
    group_labels = {}
    for row in records["train"]:
        group_labels.setdefault(row["group"], set()).add(row["label"])
    if any(len(labels) != 1 for labels in group_labels.values()):
        raise ValueError("Cross-class source groups require a custom group-stratified split")
    selected = {"train": [], "dev": [], "final": records["val"], "unused": []}
    for label, name in enumerate(names):
        rows = [r for r in records["train"] if r["label"] == label]
        if len(rows) < 4 or not any(r["label"] == label for r in records["val"]):
            raise ValueError(f"{name}: need >=4 distinct training images and >=1 val image")
        by_group = {}
        for row in rows:
            by_group.setdefault(row["group"], []).append(row)
        keys = sorted(by_group)
        random.Random(split_seed + label).shuffle(keys)
        target, dev, pool = max(2, round(len(rows) * .1)), [], []
        for key in keys:
            (dev if len(dev) < target else pool).extend(by_group[key])
        if len(pool) < 2:
            raise ValueError(f"{name}: not enough independent groups for train/dev")
        random.Random(data_seed + label).shuffle(pool)
        chosen = pool[:train_limit] if train_limit else pool
        if len(chosen) < 2:
            raise ValueError("Need at least 2 selected training images per class")
        selected["dev"].extend(dev)
        selected["train"].extend(chosen)
        selected["unused"].extend(pool[len(chosen):])
    for rows in selected.values():
        rows.sort(key=lambda r: (r["label"], r["path"]))
    result = {"version": 1, "class_names": names, "splits": selected, "data_seed": data_seed,
              "split_seed": split_seed, "final_status": "declared_unseen" if final_unseen else "historically_viewed_or_unknown",
              "source_grouping": "provided" if groups_file else "path_only_no_subject_metadata"}
    result["fingerprint"] = fingerprint(result)
    return result


def load_manifest(path):
    value = read_json(path)
    if value.get("fingerprint") != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("Frozen manifest has changed")
    return value


def verify_files(manifest, root, splits=("train", "dev")):
    root = Path(root).resolve()
    for split in splits:
        for row in manifest["splits"][split]:
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest_file(path) != row["sha256"]:
                raise ValueError(f"Dataset missing/changed/outside root: {row['path']}")


def make_dataset(manifest, root, split, image_size, augment=False):
    from torchvision import transforms
    normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    transform = transforms.Compose(([transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
        if augment else [transforms.Resize(round(image_size / .875)), transforms.CenterCrop(image_size)]) +
        [transforms.ToTensor(), normalize])
    # Top-level class is required by Windows multiprocessing spawn.
    return ManifestDataset(manifest["splits"][split], root, manifest["class_names"], transform)


# Import torch only in the data execution path; CLI help and manifests need no torch.
class ManifestDataset:
    def __init__(self, rows, root, classes, transform):
        self.samples = [(str((Path(root) / row["path"]).resolve()), row["label"]) for row in rows]
        self.classes, self.transform = classes, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def frozen_bundle(args, augment=True):
    import torch
    from torch.utils.data import DataLoader
    from datasets import DataBundle, _seed_worker
    manifest = load_manifest(args.frozen_manifest)
    verify_files(manifest, args.data_dir)
    train = make_dataset(manifest, args.data_dir, "train", args.image_size, augment)
    dev = make_dataset(manifest, args.data_dir, "dev", args.image_size)
    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
            worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(args.seed))
    return DataBundle(loader(train, True), loader(dev, False), len(manifest["class_names"]),
                      len(train), len(dev), manifest["class_names"])
