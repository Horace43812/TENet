"""MNIST/FashionMNIST/CIFAR and train/val[/test] ImageFolder loading."""
from pathlib import Path
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


SPECS = {
    'mnist': (datasets.MNIST, 1, 28, (.1307,), (.3081,), 500),
    'fashionmnist': (datasets.FashionMNIST, 1, 28, (.2860,), (.3530,), 500),
    'cifar10': (datasets.CIFAR10, 3, 32, (.4914, .4822, .4465), (.2470, .2435, .2616), 500),
    'cifar100': (datasets.CIFAR100, 3, 32, (.5071, .4867, .4408), (.2675, .2565, .2761), 50),
}


def dimensions(args):
    if args.dataset == 'imagefolder':
        return 3, args.image_size or 64
    spec = SPECS[args.dataset]
    return spec[1], args.image_size or spec[2]


def build_data(args):
    channels, size = dimensions(args)
    if args.dataset == 'imagefolder':
        mean, std = (.485, .456, .406), (.229, .224, .225)
    else:
        _, _, _, mean, std, _ = SPECS[args.dataset]
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if args.dataset == 'imagefolder':
        evaluation = transforms.Compose([transforms.Resize(round(size / .875)), transforms.CenterCrop(size)] + normalize)
        augmentation = [transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip()]
    else:
        evaluation = transforms.Compose([transforms.Resize((size, size))] + normalize)
        augmentation = [transforms.Resize((size, size))]
        if channels == 1:
            augmentation += [transforms.RandomAffine(10, translate=(.1, .1), scale=(.95, 1.05))]
        else:
            augmentation += [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()]
    training = transforms.Compose(augmentation + normalize) if args.augment else evaluation
    generator = torch.Generator().manual_seed(args.split_seed)
    monitor = 'dev' if args.eval_protocol == 'dev-test' else 'test_monitored'
    if args.smoke:
        count = args.class_count or (100 if args.dataset == 'cifar100' else 10)
        train = datasets.FakeData(32, (channels, size, size), count, training, random_offset=0)
        dev = datasets.FakeData(16, (channels, size, size), count, evaluation, random_offset=100)
        test = datasets.FakeData(16, (channels, size, size), count, evaluation, random_offset=200)
        return train, test if args.eval_protocol == 'test-monitored' else dev, test, count, {'synthetic': True, 'classes': list(range(count)), 'monitor': monitor}
    if args.dataset == 'imagefolder':
        if args.eval_protocol != 'dev-test':
            raise ValueError('ImageFolder uses val as dev and optional test; use --eval-protocol dev-test')
        root = args.data_dir
        train_source = datasets.ImageFolder(root / 'train', transform=training)
        dev_source = datasets.ImageFolder(root / 'val', transform=evaluation)
        test_source = datasets.ImageFolder(root / 'test', transform=evaluation) if (root / 'test').is_dir() else None
        if train_source.class_to_idx != dev_source.class_to_idx or (test_source and train_source.class_to_idx != test_source.class_to_idx):
            raise ValueError('train/val/test must have identical class directories')
        count = args.class_count or len(train_source.classes)
        if not 2 <= count <= len(train_source.classes):
            raise ValueError('Invalid class-count')
        train_ids = []
        for label in range(count):
            ids = [i for i, y in enumerate(train_source.targets) if y == label]
            order = torch.randperm(len(ids), generator=generator).tolist()
            train_ids.extend([ids[i] for i in order][:args.train_per_class or None])
        dev_ids = [i for i, y in enumerate(dev_source.targets) if y < count]
        test_ids = [i for i, y in enumerate(test_source.targets) if y < count] if test_source else []
        def records(source, ids):
            return [[Path(source.samples[i][0]).relative_to(root).as_posix(), source.samples[i][1]] for i in ids]
        manifest = {'classes': train_source.classes[:count], 'monitor': monitor,
                    'train': records(train_source, train_ids), 'dev': records(dev_source, dev_ids),
                    'test': records(test_source, test_ids) if test_source else []}
        return Subset(train_source, train_ids), Subset(dev_source, dev_ids), Subset(test_source, test_ids) if test_source else None, count, manifest
    dataset, _, _, _, _, default_dev = SPECS[args.dataset]
    source = dataset(args.data_dir, train=True, download=True, transform=training)
    evaluation_source = dataset(args.data_dir, train=True, download=False, transform=evaluation)
    test_source = dataset(args.data_dir, train=False, download=True, transform=evaluation)
    count = args.class_count or len(source.classes)
    if not 2 <= count <= len(source.classes):
        raise ValueError('Invalid class-count')
    dev_count = args.dev_per_class if args.dev_per_class is not None else default_dev
    if args.eval_protocol == 'test-monitored':
        dev_count = 0
    elif dev_count < 1:
        raise ValueError('dev-per-class must be positive for dev-test protocol')
    targets = torch.as_tensor(source.targets)
    train_ids, dev_ids = [], []
    for label in range(count):
        ids = (targets == label).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=generator)].tolist()
        if dev_count >= len(ids):
            raise ValueError('dev-per-class leaves no training data')
        dev_ids.extend(ids[:dev_count])
        train_ids.extend(ids[dev_count:][:args.train_per_class or None])
    test_ids = [i for i, y in enumerate(test_source.targets) if int(y) < count]
    test = Subset(test_source, test_ids)
    manifest = {'classes': source.classes[:count], 'monitor': monitor, 'split_seed': args.split_seed,
                'train': train_ids, 'dev': dev_ids, 'test': test_ids}
    dev = test if args.eval_protocol == 'test-monitored' else Subset(evaluation_source, dev_ids)
    return Subset(source, train_ids), dev, test, count, manifest
