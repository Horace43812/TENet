# Probe run `20260524_192730_cifar10_a8_L4_tr+cg`

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
  "use_edge_enhance": false,
  "use_cross_guidance": true,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_192730_cifar10_a8_L4_tr+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 208,546
- **final train acc**: 20.48%
- **final val acc**:   23.71%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 8.790 | 0.799 | 0.4474 | 0.8240 | 40.50% |
| `layers.1` | 19 | 15.019 | 0.790 | 0.3711 | 0.8770 | 41.10% |
| `layers.2` | 27 | 24.756 | 0.917 | 0.4164 | 0.7137 | 35.70% |
| `layers.3` | 35 | 26.169 | 0.748 | 0.3011 | ¡ª | 30.00% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2869 | 13.67% | 14.45% |
| 2 | 2.1590 | 20.48% | 23.71% |
