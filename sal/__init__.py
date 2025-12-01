"""
Self-Alignment Learning (SAL)
Communication-Based AI Growth

Training as dialogue, not control.

Core Components:
- CommunicationLayer: Bridge between loss and stability
- StabilityAnalyzer: Parameter classification
- EmergenceField: Coherence and novelty measurement
- PulseCascade: Semantic Game of Life
"""

__version__ = "1.0.0"
__author__ = "Aaron Liam Lee"
__email__ = "info@emergenzwerke.de"

from .communication import (
    CommunicationLayer,
    GradientStats,
    LossGuard,
)

from .stability import (
    StabilityAnalyzer,
    StabilitySpectrum,
    protect_mask,
    drift_estimator,
)

from .emergence import (
    EmergenceField,
    coherence_score,
    novelty_score,
    resonance_measure,
)

from .psc import (
    Pulse,
    Lineage,
    PulseCascade,
    emergence_select,
)

from .filters import (
    low_change_mask,
    frequency_filter,
    stability_gate,
)

from .utils import (
    cosine_similarity,
    exponential_moving_average,
    load_seed,
)

__all__ = [
    # Version
    "__version__",
    # Communication
    "CommunicationLayer",
    "GradientStats", 
    "LossGuard",
    # Stability
    "StabilityAnalyzer",
    "StabilitySpectrum",
    "protect_mask",
    "drift_estimator",
    # Emergence
    "EmergenceField",
    "coherence_score",
    "novelty_score",
    "resonance_measure",
    # PSC
    "Pulse",
    "Lineage",
    "PulseCascade",
    "emergence_select",
    # Filters
    "low_change_mask",
    "frequency_filter",
    "stability_gate",
    # Utils
    "cosine_similarity",
    "exponential_moving_average",
    "load_seed",
]
