# Probe run `20260524_192355_cifar10_a8_L4_tr+eg+cg`

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
  "use_cross_guidance": true,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_192355_cifar10_a8_L4_tr+eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 208,554
- **final train acc**: 20.08%
- **final val acc**:   23.27%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 8.772 | 0.797 | 0.5221 | 0.7784 | 40.10% |
| `layers.1` | 19 | 13.726 | 0.722 | 0.3795 | 0.8424 | 39.80% |
| `layers.2` | 27 | 23.340 | 0.864 | 0.4495 | 0.7414 | 35.75% |
| `layers.3` | 35 | 24.116 | 0.689 | 0.2963 | ¡ª | 31.05% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2918 | 13.83% | 14.90% |
| 2 | 2.1533 | 20.08% | 23.27% |
