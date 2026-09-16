"""Whole-run progress and compact best/last checkpoint summaries."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def epoch(model, loader, device, optimizer=None, smoothing=0., progress=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    totals = hits = None
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, labels, label_smoothing=smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; last complete epoch remains saved')
            if optimizer is not None:
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise RuntimeError('Non-finite gradient; last complete epoch remains saved')
                optimizer.step()
            good = output.argmax(1) == labels
            if totals is None:
                totals = torch.zeros(output.shape[1], device=device, dtype=torch.long)
                hits = torch.zeros_like(totals)
            totals += torch.bincount(labels, minlength=len(totals))
            hits += torch.bincount(labels[good], minlength=len(totals))
            count += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += good.sum().item()
            if progress is not None:
                progress.set_postfix(loss=f'{loss_sum / count:.3f}', acc=f'{100 * correct / count:.2f}%', refresh=False)
                progress.update(1)
    if count == 0:
        raise ValueError('No samples in data loader')
    balanced = (hits[totals > 0].double() / totals[totals > 0]).mean().item() * 100
    return dict(loss=loss_sum / count, accuracy_pct=100 * correct / count,
                balanced_accuracy_pct=balanced, samples=count)


def summaries(checkpoint, monitor):
    if not checkpoint:
        return None, None
    if 'best_summary' in checkpoint:
        return checkpoint['best_summary'], checkpoint.get('last_summary')
    history = checkpoint.get('history', [])
    if not history:
        return None, None
    def compact(row):
        return {'epoch': row['epoch'], 'train': row['train'], 'validation': row[monitor], 'lr': row.get('lr')}
    return compact(max(history, key=lambda r: r[monitor]['accuracy_pct'])), compact(history[-1])


def run_training(*, model, loader, evaluation, test_loader, optimizer, scheduler, shuffle,
                 device, out, epochs, start, monitor, selection_metric, best_summary, last_summary,
                 config, state_extra, record, epoch_fn=epoch, smoothing=0.):
    """Persist just best and last summaries; never an accumulating epoch list."""
    best = best_summary['validation'][selection_metric] if best_summary else -1.
    result_path = out / 'result.json'
    test_batches = len(test_loader) if test_loader is not None and not result_path.exists() else 0
    batches_per_epoch = len(loader) + len(evaluation)
    total = epochs * batches_per_epoch + test_batches
    record.update(status='running', monitor=monitor, selection_metric=selection_metric,
                  best_summary=best_summary, completed_epochs=start)
    with tqdm(total=total, initial=start * batches_per_epoch, desc='Training total', unit='batch', dynamic_ncols=True) as progress:
        for index in range(start, epochs):
            progress.set_description(f'Total | epoch {index + 1}/{epochs} train')
            training = epoch_fn(model, loader, device, optimizer, smoothing, progress=progress)
            progress.set_description(f'Total | epoch {index + 1}/{epochs} {monitor}')
            validation = epoch_fn(model, evaluation, device, progress=progress)
            last_summary = {'epoch': index + 1, 'lr': optimizer.param_groups[0]['lr'],
                            'train': training, 'validation': validation}
            improved = validation[selection_metric] > best
            if improved:
                best, best_summary = validation[selection_metric], last_summary
            scheduler.step()
            state = {**state_extra, 'config': config, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': index + 1,
                     'best_summary': best_summary, 'last_summary': last_summary, 'best_monitor': best,
                     'best_dev': best, 'rng': torch.get_rng_state(), 'shuffle_rng': shuffle.get_state(),
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
            if improved:
                save_checkpoint(out / 'best.pth', state)
            save_checkpoint(out / 'last.pth', state)
            write_json(out / 'summary.json', {'monitor': monitor, 'selection_metric': selection_metric,
                                             'best': best_summary, 'completed_epochs': index + 1})
            record.update(best_summary=best_summary, completed_epochs=index + 1)
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
        else:
            # Dev-selected best checkpoint, not whichever epoch happens to finish last.
            selected = torch.load(out / 'best.pth', map_location='cpu', weights_only=False)
            model.load_state_dict(selected['model'])
            progress.set_description(f'Total | best epoch {best_summary["epoch"]} test')
            testing = epoch_fn(model, test_loader, device, progress=progress) if test_loader is not None else None
            result = {'parameters': sum(p.numel() for p in model.parameters()),
                      'model_version': config.get('model_version', 'tiny_v1'),
                      'normalization': getattr(model, 'resolved_options', {}).get('normalization', config.get('normalization')),
                      'dataset': config.get('dataset', 'mnist'), 'seed': config['seed'], 'epoch': epochs,
                      'checkpoint': 'best.pth', 'monitor': monitor, 'selection_metric': selection_metric,
                      'best_monitor_pct': best, 'best_epoch': best_summary['epoch'],
                      'best_monitor': best_summary['validation'], 'test': testing,
                      'synthetic': config['smoke'], 'torch': torch.__version__, 'device': str(device)}
            write_json(result_path, result)
    record.update(status='complete', completed_epochs=epochs, best_summary=best_summary,
                  test=result.get('test'), test_checkpoint=result.get('checkpoint'))
    print(json.dumps(result, indent=2))
    print(f'Training record: {record.folder / "training_records.md"}')
