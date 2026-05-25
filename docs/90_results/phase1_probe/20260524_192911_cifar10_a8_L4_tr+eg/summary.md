# Probe run `20260524_192911_cifar10_a8_L4_tr+eg`

## Config

```json
{
  "dataset": "cifar10",
  "alpha": 8,
  "num_layers": 4,
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
  "use_cross_guidance": false,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_192911_cifar10_a8_L4_tr+eg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 203,314
- **final train acc**: 18.28%
- **final val acc**:   23.51%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 9.919 | 0.902 | 0.5477 | 0.8637 | 41.00% |
| `layers.1` | 19 | 15.340 | 0.807 | 0.4616 | 0.8273 | 40.50% |
| `layers.2` | 27 | 22.266 | 0.825 | 0.3880 | 0.7584 | 35.05% |
| `layers.3` | 35 | 21.029 | 0.601 | 0.2939 | ¡ª | 30.45% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2969 | 13.23% | 12.05% |
| 2 | 2.1833 | 18.28% | 23.51% |
