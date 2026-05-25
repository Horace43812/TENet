# Probe run `20260524_190847_mnist_a1_L2_tr+eg+cg`

## Config

```json
{
  "dataset": "mnist",
  "alpha": 1,
  "num_layers": 2,
  "epochs": 2,
  "batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "seed": 42,
  "probe_batches": 4,
  "probe_epochs": 10,
  "probe_lr": 0.01,
  "use_trace_branch": true,
  "use_edge_enhance": true,
  "use_cross_guidance": true,
  "cpu": true,
  "checkpoint": "",
  "no_train": false,
  "no_save_model": false,
  "out_root": "",
  "run_id": "20260524_190847_mnist_a1_L2_tr+eg+cg",
  "data_dir": "./data"
}
```

- **#params**: 8,543
- **final train acc**: 92.28%
- **final val acc**:   94.66%

## Per-layer information-flow metrics

| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |
|---|---|---|---|---|---|---|
| `layer1` | 2 | 1.621 | 0.810 | 0.9447 | 0.7941 | 90.75% |
| `layer2` | 3 | 2.964 | 0.988 | 0.7327 | ¡ª | 93.25% |

## Training history

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 0.9022 | 80.37% | 92.57% |
| 2 | 0.5394 | 92.28% | 94.66% |
