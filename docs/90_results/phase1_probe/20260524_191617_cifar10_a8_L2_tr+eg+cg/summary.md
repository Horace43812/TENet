# Probe run `20260524_191617_cifar10_a8_L2_tr+eg+cg`

## Config

```json
{
  "dataset": "cifar10",
  "alpha": 8,
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
  "run_id": "20260524_191617_cifar10_a8_L2_tr+eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 261,674
- **final train acc**: 30.44%
- **final val acc**:   36.94%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 8.378 | 0.762 | 0.6147 | 0.8475 | 38.65% |
| `layers.1` | 19 | 13.215 | 0.696 | 0.5862 | ¡ª | 45.05% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2122 | 17.73% | 23.03% |
| 2 | 1.9365 | 30.44% | 36.94% |
