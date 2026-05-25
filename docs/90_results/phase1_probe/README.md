# Phase-1 Probe Results

This directory archives every TENet information-flow probe run.

Each run lands in a sub-directory named:

```
<YYYYmmdd_HHMMSS>_<dataset>_a<alpha>_L<num_layers>_<modules>/
    metrics.json    raw per-layer numbers (effective rank, CKA, probe acc, ...)
    summary.md      human-readable table + training history
    model.pt        trained model weights
    phi_*.pt        sampled phi(v_i, v_j) heatmaps (one per GraphifiedLayer)
```

## How to add a new run

From the `Code/` directory:

```bash
# Smallest sanity check — MNIST, 1-channel, alpha=1 (~2.4k params reproduction)
python -m analysis.run_probe --dataset mnist --alpha 1 --epochs 5

# CIFAR-10 bandwidth sweep
python -m analysis.run_probe --dataset cifar10 --alpha 1  --num-layers 2 --epochs 10
python -m analysis.run_probe --dataset cifar10 --alpha 8  --num-layers 2 --epochs 10
python -m analysis.run_probe --dataset cifar10 --alpha 32 --num-layers 2 --epochs 10

# CIFAR-10 depth sweep at alpha=8
python -m analysis.run_probe --dataset cifar10 --alpha 8 --num-layers 4 --epochs 10
```

After each run completes, update [../../10_phase1_mechanism/02_findings.md](../../10_phase1_mechanism/02_findings.md)
with the new row(s) so the analysis stays current.
