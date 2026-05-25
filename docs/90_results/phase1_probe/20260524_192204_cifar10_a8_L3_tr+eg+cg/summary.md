# Probe run `20260524_192204_cifar10_a8_L3_tr+eg+cg`

## Config

```json
{
  "dataset": "cifar10",
  "alpha": 8,
  "num_layers": 3,
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
  "run_id": "20260524_192204_cifar10_a8_L3_tr+eg+cg",
  "data_dir": "./data",
  "max_train_batches": 50,
  "log_every": 25
}
```

- **#params**: 165,302
- **final train acc**: 22.27%
- **final val acc**:   26.74%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layers.0` | 11 | 8.214 | 0.747 | 0.5679 | 0.7936 | 38.75% |
| `layers.1` | 19 | 14.402 | 0.758 | 0.5686 | 0.8650 | 42.00% |
| `layers.2` | 27 | 22.677 | 0.840 | 0.5357 | ¡ª | 38.85% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2951 | 12.94% | 13.55% |
| 2 | 2.1509 | 22.27% | 26.74% |
