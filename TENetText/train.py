"""Train TENet/TextCNN/mean classifiers using identical text preprocessing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from text_data import prepare
from text_model import TextClassifier, matched_width

# Reuse the existing checkpoint/progress/record machinery without altering image runs.
TINY = Path(__file__).resolve().parents[1] / 'TENetTiny'
sys.path.insert(0, str(TINY))
from runtime import run_training, summaries, write_json, epoch
from records import RunRecord, DEFAULT_RECORDS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('agnews', 'csv'), default='agnews')
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data/agnews')
    p.add_argument('--model-version', choices=('tenet', 'cnn', 'mean'), default='tenet')
    p.add_argument('--tokenizer', choices=('word', 'char'), default='word')
    p.add_argument('--vocab-size', type=int, default=20000, help='Cap including PAD/UNK; built only from train partition')
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--embedding-dim', type=int, default=16)
    p.add_argument('--hidden-dim', type=int, default=16)
    p.add_argument('--target-backbone-params', type=int, help='Auto-select hidden width, excluding embedding parameters')
    p.add_argument('--normalization', choices=('raw', 'centered', 'standardized'), default=None)
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--dev-fraction', type=float, default=.1)
    p.add_argument('--split-seed', type=int, default=20260919)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--records-dir', type=Path, default=DEFAULT_RECORDS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Synthetic text only; no download')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.max_length, args.embedding_dim, args.hidden_dim, args.epochs, args.batch_size, args.lr, args.threads) <= 0 or args.vocab_size < 3:
        p.error('Dimensions, epochs, batch size, threads and lr must be positive; vocab-size >=3')
    if min(args.seed, args.split_seed, args.workers, args.weight_decay) < 0 or not 0 < args.dev_fraction < 1 or not 0 <= args.dropout < 1:
        p.error('Invalid seed/count/weight decay/dev fraction/dropout')
    if args.target_backbone_params is not None and args.target_backbone_params <= 0:
        p.error('Parameter target must be positive')
    if args.model_version != 'tenet' and args.normalization is not None:
        p.error('--normalization applies only to TENet')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ('output_dir', 'records_dir', 'resume', 'workers', 'threads', 'cpu')}
    fingerprint = hashlib.sha256()
    for path in [Path(__file__), Path(__file__).with_name('text_data.py'), Path(__file__).with_name('text_model.py'),
                 TINY / 'runtime.py', TINY / 'records.py']:
        fingerprint.update(path.read_bytes())
    config['source_sha256'] = fingerprint.hexdigest()
    config['task'] = 'text_classification'
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    out = args.output_dir or Path(__file__).resolve().parent / 'runs' / f'{args.dataset}_{args.model_version}_s{args.seed}_{key}'
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            p.error('Resume requires identical code/configuration, including original --epochs')
    elif out.exists() and any(out.iterdir()):
        p.error('Output is nonempty; use a new directory or --resume with original configuration')
    with RunRecord(args.records_dir, out, config) as record:
        print(f'Preparing {args.dataset}: {args.data_dir.resolve()} (download if missing, tokenize, freeze split)...', flush=True)
        train, dev, test, vocab, split = prepare(args)
        if checkpoint and (checkpoint['split'] != split or checkpoint['vocab'] != vocab):
            raise ValueError('Dataset/split/vocabulary changed since checkpoint')
        classes = len(split['classes'])
        width = (matched_width(args.model_version, args.embedding_dim, classes, args.target_backbone_params)
                 if args.target_backbone_params else args.hidden_dim)
        kwargs = dict(vocab_size=len(vocab) + 2, num_classes=classes, model_version=args.model_version,
                      embedding_dim=args.embedding_dim, hidden_dim=width,
                      normalization=args.normalization or 'raw', dropout=args.dropout)
        torch.manual_seed(args.seed)
        model = TextClassifier(**kwargs).to(device)
        budget = model.parameter_counts()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: .1 + .9 * (1 + math.cos(math.pi * min(e, args.epochs) / args.epochs)) / 2)
        shuffle = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=shuffle, num_workers=args.workers)
        evaluation = DataLoader(dev, batch_size=args.batch_size, num_workers=args.workers)
        test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)
        start = 0
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
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'config.json', dict(training=config, model=kwargs, resolved=model.resolved_options, parameter_counts=budget))
        write_json(out / 'vocab.json', dict(tokens=vocab, pad_id=0, unk_id=1, tokenizer=args.tokenizer))
        write_json(out / 'split.json', split)
        record.update(parameters=budget['total'], parameter_counts=budget, model_config=kwargs,
                      resolved_model=model.resolved_options, normalization=model.resolved_options['normalization'],
                      dataset_sizes=dict(train=len(train), dev=len(dev), test=len(test)),
                      device=str(device), torch=torch.__version__, workers=args.workers, threads=args.threads)
        print(f'Device: {device} | Model: {args.model_version} | Parameters: {budget} | Hidden: {width}', flush=True)
        print(f'train/dev/test: {len(train)}/{len(dev)}/{len(test)} | Vocabulary: {len(vocab) + 2} | Output: {out.resolve()}', flush=True)
        best, last = summaries(checkpoint, 'dev')
        run_training(model=model, loader=loader, evaluation=evaluation, test_loader=test_loader,
                     optimizer=optimizer, scheduler=scheduler, shuffle=shuffle, device=device, out=out,
                     epochs=args.epochs, start=start, monitor='dev', selection_metric='accuracy_pct',
                     best_summary=best, last_summary=last, config=config,
                     state_extra=dict(model_kwargs=kwargs, vocab=vocab, split=split, parameter_counts=budget),
                     record=record, epoch_fn=epoch)


if __name__ == '__main__':
    main()
