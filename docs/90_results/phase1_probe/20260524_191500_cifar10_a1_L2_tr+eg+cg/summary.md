# Probe run `20260524_191500_cifar10_a1_L2_tr+eg+cg`

## Config

```json
{
  "dataset": "cifar10",
  "alpha": 1,
  "num_layers": 2,
  "epochs": 2,
  "batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "seed": 42,
  "probe_batches": 4,
  "probe_epochs": 15,
  "probe_lr": 0.01,
  "use_trace_branch": true,
  "use_edge_enhance": true,
  "use_cross_guidance": true,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_191500_cifar10_a1_L2_tr+eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 67,067
- **final train acc**: 28.88%
- **final val acc**:   30.59%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 4 | 3.565 | 0.891 | 0.4549 | 0.7615 | 35.05% |
| `layers.1` | 5 | 3.776 | 0.755 | 0.5900 | ¡ª | 40.65% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2385 | 16.09% | 24.69% |
| 2 | 1.9903 | 28.88% | 30.59% |
