"""AG News/local CSV data, training-only vocabulary, reproducible dev split."""
from collections import Counter
import csv
import hashlib
from pathlib import Path
import random
import re
import time
import urllib.request

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

BASE_URL = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'


def read_rows(path, agnews=False):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if not agnews and next(reader, None) != ['label', 'text']:
            raise ValueError('Local CSV must have header: label,text')
        for row in reader:
            if len(row) != (3 if agnews else 2):
                raise ValueError(f'Invalid CSV row in {path}: expected label and text fields')
            if not row[0] or not ' '.join(row[1:]).strip():
                raise ValueError('Empty label/text in CSV')
            rows.append((row[0], ' '.join(row[1:])))
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def download(path, expected_rows):
    """Bounded network waits, visible progress, validation before atomic publish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(BASE_URL + path.name, timeout=30) as source:
                size = int(source.headers.get('Content-Length', 0))
                with temporary.open('wb') as target, tqdm(total=size or None, unit='B', unit_scale=True, desc=path.name) as bar:
                    while True:
                        block = source.read(65536)
                        if not block:
                            break
                        target.write(block)
                        bar.update(len(block))
                if size and temporary.stat().st_size != size:
                    raise ValueError('Incomplete download')
            rows = read_rows(temporary, agnews=True)
            if len(rows) != expected_rows or set(label for label, _ in rows) != {'1', '2', '3', '4'}:
                raise ValueError('AG News download failed row/class validation')
            temporary.replace(path)
            return
        except Exception as error:
            if attempt == 2:
                raise RuntimeError(f'Download failed: {path.name}. Place the official CSV in {path.parent} and retry.') from error
            print(f'Download interrupted ({error}); retry {attempt + 1}/2', flush=True)
            time.sleep(2)


def tokenize(text, mode):
    return list(text.lower().strip()) if mode == 'char' else re.findall(r"\w+|[^\w\s]", text.lower())


def prepare(args):
    paths = [args.data_dir / name for name in ('train.csv', 'test.csv')]
    if args.smoke:
        training = [(str(c), f'topic{c} sample {i} text') for c in range(4) for i in range(12)]
        testing = [(str(c), f'topic{c} heldout example') for c in range(4) for _ in range(2)]
        hashes = {'synthetic': True}
    else:
        if args.dataset == 'agnews':
            for path, count in zip(paths, (120000, 7600)):
                if not path.exists():
                    download(path, count)
        training, testing = [read_rows(path, agnews=args.dataset == 'agnews') for path in paths]
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    classes = sorted({label for label, _ in training})
    if len(classes) < 2 or any(label not in classes for label, _ in testing):
        raise ValueError('Need >=2 training classes; test labels must occur in training')
    mapping = {label: i for i, label in enumerate(classes)}
    rng = random.Random(args.split_seed)
    train_ids, dev_ids = [], []
    for label in classes:
        ids = [i for i, row in enumerate(training) if row[0] == label]
        if len(ids) < 2:
            raise ValueError('Each class needs >=2 training examples for dev split')
        rng.shuffle(ids)
        count = min(len(ids) - 1, max(1, round(len(ids) * args.dev_fraction)))
        dev_ids.extend(ids[:count])
        train_ids.extend(ids[count:])
    train_ids.sort()
    dev_ids.sort()
    counts = Counter(token for i in train_ids for token in tokenize(training[i][1], args.tokenizer))
    words = sorted(counts, key=lambda token: (-counts[token], token))[:args.vocab_size - 2]
    vocab = {word: i + 2 for i, word in enumerate(words)}
    def encode(rows):
        tokens = torch.zeros((len(rows), args.max_length), dtype=torch.long)
        labels = torch.empty(len(rows), dtype=torch.long)
        for i, (label, text) in enumerate(rows):
            ids = [vocab.get(word, 1) for word in tokenize(text, args.tokenizer)[:args.max_length]] or [1]
            tokens[i, :len(ids)] = torch.tensor(ids)
            labels[i] = mapping[label]
        return TensorDataset(tokens, labels)
    split = dict(train=train_ids, dev=dev_ids, hashes=hashes, classes=classes, monitor='dev')
    return (encode([training[i] for i in train_ids]), encode([training[i] for i in dev_ids]),
            encode(testing), vocab, split)
