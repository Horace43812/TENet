"""Shared hook infrastructure for TENet mechanism analysis.

Provides:
    ActivationCollector - context manager that captures every GraphifiedLayer's
                          (x_topo, x_content) outputs.
    iter_graphified_layers - yields (name, module) pairs for every GraphifiedLayer.

Works with MNIST_TENet_Current / MNIST_TENet_Legacy (layer1/layer2 attributes)
and TENet_RGB32 / TENet_RGB256 (self.layers ModuleList).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn


def _is_graphified(module: nn.Module) -> bool:
    return module.__class__.__name__ == "GraphifiedLayer"


def iter_graphified_layers(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if _is_graphified(module):
            yield name, module


class ActivationCollector:
    """Capture per-GraphifiedLayer topology / content outputs during forward.

    Usage:
        collector = ActivationCollector(model)
        with collector:
            _ = model(x)
        records = collector.records  # list of dicts, one per layer
    """

    def __init__(self, model: nn.Module, detach: bool = True, to_cpu: bool = True):
        self.model = model
        self.detach = detach
        self.to_cpu = to_cpu
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._buffer: List[Dict] = []
        self._layer_names: List[str] = []

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            entry: Dict = {"name": name}
            if isinstance(output, tuple):
                entry["topo"] = self._maybe_detach(output[0])
                if len(output) > 1:
                    entry["content"] = self._maybe_detach(output[1])
            else:
                entry["topo"] = self._maybe_detach(output)
            self._buffer.append(entry)
        return hook

    def _maybe_detach(self, t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return t
        if self.detach:
            t = t.detach()
        if self.to_cpu:
            t = t.cpu()
        return t

    def __enter__(self):
        self._buffer.clear()
        self._layer_names.clear()
        for name, module in iter_graphified_layers(self.model):
            self._layer_names.append(name)
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def records(self) -> List[Dict]:
        return list(self._buffer)

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_names)


@contextmanager
def capture_activations(model: nn.Module, **kwargs):
    collector = ActivationCollector(model, **kwargs)
    with collector:
        yield collector
