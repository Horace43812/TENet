# Probe run `20260524_192544_cifar10_a8_L4_eg+cg`

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
  "use_trace_branch": false,
  "use_edge_enhance": true,
  "use_cross_guidance": true,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_192544_cifar10_a8_L4_eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 206,670
- **final train acc**: 19.12%
- **final val acc**:   21.84%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 9.830 | 0.894 | 0.5884 | 0.8656 | 36.45% |
| `layers.1` | 19 | 17.747 | 0.934 | 0.6284 | 0.7209 | 35.00% |
| `layers.2` | 27 | 26.239 | 0.972 | 0.4883 | 0.5777 | 27.70% |
| `layers.3` | 35 | 32.349 | 0.924 | 0.3811 | ¡ª | 23.90% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2855 | 12.39% | 16.87% |
| 2 | 2.1651 | 19.12% | 21.84% |
