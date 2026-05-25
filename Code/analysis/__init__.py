"""TENet / RGA mechanism-analysis toolkit.

Modules:
    effective_rank   - tool 1: bandwidth probe via SVD entropy
    cka              - tool 2: linear CKA between layer activations
    input_grad       - tool 3: input-gradient saliency / cross-subtree leakage
    linear_probe     - tool 4: per-layer task-info via frozen linear probe
    phi_visualize    - tool 5: visualise the learned edge function phi
    hooks            - shared infrastructure for capturing intermediate activations
"""

from .effective_rank import effective_rank, effective_rank_per_channel
from .cka import linear_cka
from .hooks import ActivationCollector, iter_graphified_layers

__all__ = [
    "effective_rank",
    "effective_rank_per_channel",
    "linear_cka",
    "ActivationCollector",
    "iter_graphified_layers",
]
