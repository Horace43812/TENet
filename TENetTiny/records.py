"""One summary per experiment directory; SQLite serializes concurrent writers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


DEFAULT_RECORDS = Path(__file__).resolve().parent / 'records'


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def export(connection, folder):
    rows = [json.loads(row[0]) for row in connection.execute('SELECT body FROM runs ORDER BY started')]
    lines = ['# TENetTiny training records', '', 'One row per experiment. Resume updates the same row; no epoch history.', '',
             '| Started (UTC) | Dataset | Version | Norm | Parameters | Status | Best metric (%) | Epoch | Test (%) / checkpoint | Configuration | Output |',
             '| --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |']
    def clean(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for row in rows:
        config = row.get('config', {})
        best = row.get('best_summary') or {}
        validation = best.get('validation', {})
        metric = row.get('selection_metric', 'accuracy_pct')
        value = validation.get(metric)
        best_text = f"{row.get('monitor', 'dev')}/{metric}: {value:.4f}" if value is not None else '-'
        test = row.get('test') or {}
        test_value = test.get(metric, test.get('accuracy_pct'))
        test_text = f"{test_value:.4f} / {row.get('test_checkpoint', '-')}" if test_value is not None else '-'
        settings = {'training': config, 'model': row.get('model_config'), 'resolved_model': row.get('resolved_model'),
                    'execution': {key: row.get(key) for key in ('device', 'torch', 'workers', 'threads')}}
        cells = [row['started'], config.get('dataset', 'mnist'), config.get('model_version', 'tiny_v1'),
                 row.get('normalization', config.get('normalization')), row.get('parameters', '-'), row['status'],
                 best_text, best.get('epoch', '-'), test_text, json.dumps(settings, ensure_ascii=False, separators=(',', ':')), row['output_dir']]
        lines.append('| ' + ' | '.join(clean(cell) for cell in cells) + ' |')
    for name, value in [('training_records.json', json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)),
                        ('training_records.md', '\n'.join(lines) + '\n')]:
        destination = folder / name
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(value, encoding='utf-8')
        temporary.replace(destination)


class RunRecord:
    def __init__(self, folder, output, config):
        self.folder, self.output = Path(folder), str(Path(output).resolve())
        self.key = hashlib.sha256(self.output.encode()).hexdigest()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.update(config=config, status='preparing', error=None)

    def update(self, **changes):
        with sqlite3.connect(self.folder / 'training_records.sqlite3', timeout=30) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started TEXT NOT NULL, body TEXT NOT NULL)')
            connection.execute('BEGIN IMMEDIATE')
            saved = connection.execute('SELECT body FROM runs WHERE id=?', (self.key,)).fetchone()
            row = json.loads(saved[0]) if saved else {'id': self.key, 'started': timestamp(), 'output_dir': self.output}
            row.update(changes)
            row['updated'] = timestamp()
            connection.execute('INSERT OR REPLACE INTO runs VALUES (?, ?, ?)',
                               (self.key, row['started'], json.dumps(row, ensure_ascii=False, allow_nan=False)))
            export(connection, self.folder)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.update(status='interrupted' if issubclass(kind, KeyboardInterrupt) else 'failed',
                        error=f'{kind.__name__}: {value}')
        return False
