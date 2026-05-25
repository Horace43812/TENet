# Probe run `20260524_191844_cifar10_a32_L2_tr+eg+cg`

## Config

```json
{
  "dataset": "cifar10",
  "alpha": 32,
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
  "run_id": "20260524_191844_cifar10_a32_L2_tr+eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 1,088,114
- **final train acc**: 36.30%
- **final val acc**:   43.05%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 35 | 23.144 | 0.661 | 0.5804 | 0.9181 | 43.60% |
| `layers.1` | 67 | 53.522 | 0.799 | 0.5805 | ¡ª | 48.45% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.1713 | 20.97% | 31.43% |
| 2 | 1.8160 | 36.30% | 43.05% |
