"""TENetTiny training; all architecture controls work with every dataset."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_data, dimensions, class_count_hint, EMNIST_CLASSES
from model import TENetTiny, DEFAULTS
from runtime import epoch, write_json, summaries, run_training
from records import RunRecord, DEFAULT_RECORDS


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = p.add_argument_group('Dataset and fixed split')
    data.add_argument('--dataset', choices=('mnist', 'fashionmnist', 'emnist', 'kmnist', 'k49', 'cifar10', 'cifar100', 'imagefolder'), default='mnist')
    data.add_argument('--emnist-split', choices=tuple(EMNIST_CLASSES), default='balanced')
    data.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'Code/data')
    data.add_argument('--image-size', type=int, help='Default: 28 for grayscale, 32 for CIFAR, 64 for ImageFolder')
    data.add_argument('--class-count', type=int, default=0, help='0=all; otherwise first N classes')
    data.add_argument('--train-per-class', type=int, default=0, help='0=all training candidates')
    data.add_argument('--dev-per-class', type=int, help='Built-in dev size per class: 500, or 50 for CIFAR100')
    data.add_argument('--split-seed', type=int, default=20260913)
    data.add_argument('--eval-protocol', choices=('dev-test', 'test-monitored'), default='dev-test', help='test-monitored uses all official train and monitors official test each epoch')
    data.add_argument('--augment', action=argparse.BooleanOptionalAction, default=False)
    data.add_argument('--no-aug', dest='augment', action='store_false', help=argparse.SUPPRESS)
    model = p.add_argument_group('Model: unspecified controls inherit the selected version')
    model.add_argument('--model-version', choices=tuple(DEFAULTS), default='current')
    model.add_argument('--normalization', '--graph-normalization', choices=('raw', 'centered', 'standardized'))
    model.add_argument('--window-layout', choices=('historical', 'corrected'))
    model.add_argument('--layers', type=int, default=2)
    model.add_argument('--alpha', type=int, default=1)
    model.add_argument('--head', choices=('linear', 'linear-relu', 'mlp'))
    model.add_argument('--head-hidden', type=int, default=48)
    model.add_argument('--classifier-dropout', type=float)
    aliases = {'trace': '--trace-branch', 'edge': '--edge-enhance', 'directed': '--directed-flow',
               'cross': '--cross-guidance', 'chebyshev': '--use-chebyshev', 'spatial': '--use-spatial-bridge'}
    for name, description in (
        ('learnable-phi', 'Learnable edge residual'), ('trace', 'Trace supplement branch'),
        ('edge', 'Squared-difference and product edge terms'), ('directed', 'Signed incidence flow'),
        ('cross', 'Bidirectional content/topology guidance; enables content stream'),
        ('chebyshev', 'L/L-squared matrix mixture'), ('spatial', 'Dilated depthwise spatial bridge'),
        ('content', 'Content stream; disable cross before disabling this'),
        ('main-gate', 'Historical main-branch gate, active when trace is enabled')):
        names = ['--' + name] + ([aliases[name]] if name in aliases else [])
        model.add_argument(*names, action=argparse.BooleanOptionalAction, default=None, help=description)
    model.add_argument('--trace-init', type=float, default=.1)
    model.add_argument('--phi-temp-init', type=float, default=1.)
    model.add_argument('--phi-residual-init', type=float, default=.1)
    training = p.add_argument_group('Training and checkpoints')
    training.add_argument('--epochs', type=int, default=50)
    training.add_argument('--batch-size', type=int, default=128)
    training.add_argument('--lr', type=float, default=1e-3)
    training.add_argument('--optimizer', choices=('adam', 'adamw'), default='adam')
    training.add_argument('--weight-decay', type=float, default=0.)
    training.add_argument('--label-smoothing', type=float, default=0.)
    training.add_argument('--trace-lr-mult', type=float, default=1.)
    training.add_argument('--scheduler', choices=('none', 'cosine', 'warmup-cosine'), default='cosine')
    training.add_argument('--warmup-epochs', type=int, default=3)
    training.add_argument('--min-lr-ratio', type=float, default=.1)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--workers', type=int, default=0)
    training.add_argument('--threads', type=int, default=4)
    training.add_argument('--cpu', action='store_true')
    training.add_argument('--output-dir', type=Path)
    training.add_argument('--records-dir', type=Path, default=DEFAULT_RECORDS)
    training.add_argument('--selection-metric', choices=('auto', 'accuracy_pct', 'balanced_accuracy_pct'), default='auto', help='auto uses balanced accuracy for k49, Top-1 elsewhere')
    training.add_argument('--resume', action='store_true')
    training.add_argument('--smoke', action='store_true', help='Synthetic images; no dataset download')
    training.add_argument('--describe', action='store_true', help='Print resolved model and parameter budget, no dataset access')
    return p


def model_kwargs(args, num_classes):
    channels, size = dimensions(args)
    options = {key: getattr(args, key) for key in DEFAULTS[args.model_version]}
    return dict(model_version=args.model_version, in_channels=channels, image_size=size,
                num_classes=num_classes, layers=args.layers, alpha=args.alpha, head_hidden=args.head_hidden,
                trace_init=args.trace_init, phi_temp_init=args.phi_temp_init,
                phi_residual_init=args.phi_residual_init, **options)


PREVIOUS_SOURCES = {
    '5c9a19e7e56621f9e9ac6d7e666026056fb9fb8635c73808c0f83fa7cf32cae5',
    '1922892f99b7a6ff97caec4528bf32b66849ff28dca8a6d6297687fd6e0a9530',
}


def compatible_resume(previous, current):
    if previous == current:
        return True
    if previous.get('source_sha256') not in PREVIOUS_SOURCES:
        return False
    migrated = dict(current)
    if migrated.pop('emnist_split') != 'balanced' or migrated.pop('selection_metric') != 'auto':
        return False
    migrated['source_sha256'] = previous['source_sha256']
    return previous == migrated


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.lr, args.threads, args.trace_lr_mult) <= 0:
        p.error('Epochs, batch size, lr, threads and trace-lr-mult must be positive')
    if min(args.seed, args.split_seed, args.workers, args.weight_decay, args.warmup_epochs, args.class_count, args.train_per_class) < 0:
        p.error('Counts, seeds and weight-decay must be nonnegative')
    if not 0 <= args.label_smoothing < 1 or not 0 <= args.min_lr_ratio <= 1:
        p.error('Invalid smoothing or min-lr-ratio')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    estimated_classes = class_count_hint(args)
    preview = TENetTiny(**model_kwargs(args, estimated_classes))
    if args.batch_size == 1 and dimensions(args)[1] // 2 ** args.layers == 1:
        p.error('BatchNorm at 1x1 requires batch-size >= 2')
    if args.describe:
        print(json.dumps({'model': model_kwargs(args, estimated_classes), 'resolved': preview.resolved_options,
            'parameters': sum(p.numel() for p in preview.parameters()),
            'note': 'ImageFolder class count estimated; pass --class-count for exact budget'}, indent=2))
        return
    del preview
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('output_dir', 'records_dir', 'resume', 'workers', 'threads', 'cpu', 'describe')}
    # Detect accidental architecture/training source edits when resuming new-format runs.
    source_hash = hashlib.sha256()
    for name in ('model.py', 'data.py', 'character_data.py', 'train.py', 'runtime.py', 'records.py', 'tiny_v1.py'):
        source_hash.update(Path(__file__).with_name(name).read_bytes())
    config['source_sha256'] = source_hash.hexdigest()
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    norm = args.normalization or DEFAULTS[args.model_version]['normalization']
    out = args.output_dir or Path(__file__).resolve().parent / 'runs' / f'{args.dataset}_{args.model_version}_{norm}_s{args.seed}_{key}'
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
        if not compatible_resume(checkpoint['config'], config):
            p.error('Resume requires original configuration and code; specify the original --epochs and --output-dir')
    elif out.exists() and any(out.iterdir()):
        p.error('Output is not empty; use --resume or a new --output-dir')
    with RunRecord(args.records_dir, out, config) as record:
        execute(args, config, checkpoint, out, norm, device, record)


def execute(args, config, checkpoint, out, norm, device, record):
    train, dev, test, classes, split = build_data(args)
    if checkpoint and checkpoint['split'] != split:
        raise ValueError('Dataset paths, labels or split changed')
    out.mkdir(parents=True, exist_ok=True)
    kwargs = model_kwargs(args, classes)
    # Data preparation must not change model initialization across augmentation choices.
    torch.manual_seed(args.seed)
    model = TENetTiny(**kwargs).to(device)
    parameters = sum(v.numel() for v in model.parameters())
    main_params, trace_params = [], []
    for name, parameter in model.named_parameters():
        (trace_params if 'trace_' in name else main_params).append(parameter)
    groups = [{'params': main_params, 'lr': args.lr}]
    if trace_params:
        groups.append({'params': trace_params, 'lr': args.lr * args.trace_lr_mult})
    optimizer_class = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
    optimizer = optimizer_class(groups, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * min(e, args.epochs) / args.epochs)) / 2)
    elif args.scheduler == 'warmup-cosine':
        def schedule(e):
            if e < args.warmup_epochs:
                return (e + 1) / max(1, args.warmup_epochs)
            progress = min(1., (e - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs))
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.)
    shuffle = torch.Generator().manual_seed(args.seed)
    # No persistent workers; worker seeds are regenerated from the saved generator each epoch.
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=shuffle, num_workers=args.workers,
                        drop_last=len(train) % args.batch_size == 1 and dimensions(args)[1] // 2 ** args.layers == 1)
    evaluation = DataLoader(dev, batch_size=args.batch_size, num_workers=args.workers)
    start = 0
    monitor = split['monitor']
    best_summary, last_summary = summaries(checkpoint, monitor)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['epoch']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['rng'])
        shuffle.set_state(checkpoint['shuffle_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    write_json(out / 'config.json', {'training': config, 'model': kwargs, 'resolved': model.resolved_options, 'parameters': parameters})
    write_json(out / 'split.json', split)
    monitor = split['monitor']
    print(f'Device: {device} | Model: {args.model_version} | Parameters: {parameters:,} | Graph norm: {norm}', flush=True)
    print(f'Options: {model.resolved_options}', flush=True)
    print(f'Output: {out.resolve()} | train/{monitor}/test: {len(train)}/{len(dev)}/{len(test) if test is not None else 0}', flush=True)
    record.update(parameters=parameters, model_config=kwargs, resolved_model=model.resolved_options, normalization=norm,
                  dataset_sizes={'train': len(train), 'monitor': len(dev), 'test': len(test) if test is not None else 0},
                  device=str(device), torch=torch.__version__, workers=args.workers, threads=args.threads)
    metric = args.selection_metric
    if metric == 'auto':
        metric = 'balanced_accuracy_pct' if args.dataset == 'k49' else 'accuracy_pct'
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers) if test is not None else None
    run_training(model=model, loader=loader, evaluation=evaluation, test_loader=test_loader, optimizer=optimizer,
                 scheduler=scheduler, shuffle=shuffle, device=device, out=out, epochs=args.epochs, start=start,
                 monitor=monitor, selection_metric=metric, best_summary=best_summary, last_summary=last_summary,
                 config=config, state_extra={'model_kwargs': kwargs, 'split': split}, record=record,
                 epoch_fn=epoch, smoothing=args.label_smoothing)


if __name__ == '__main__':
    main()
