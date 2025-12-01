"""
SAL Stability Module

Analyzes and classifies parameter stability.
Protects identity while enabling growth.

Stability is not rigidity — it's coherent persistence.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StabilityState(Enum):
    """The three states of parameter stability."""
    PROTECTED = "protected"   # Identity core - never overwritten
    NEUTRAL = "neutral"       # Adaptive zone - updated with care
    VOLATILE = "volatile"     # Learning edge - open to change


@dataclass
class StabilitySpectrum:
    """
    Distribution of parameters across stability states.
    
    A healthy model has:
    - ~10-15% protected (identity core)
    - ~65-75% neutral (adaptive capacity)
    - ~15-20% volatile (learning edge)
    """
    
    protected: float  # Percentage of protected parameters
    neutral: float    # Percentage of neutral parameters
    volatile: float   # Percentage of volatile parameters
    
    def __post_init__(self):
        """Validate percentages sum to ~100%."""
        total = self.protected + self.neutral + self.volatile
        if abs(total - 100.0) > 0.1:
            # Normalize
            self.protected = (self.protected / total) * 100
            self.neutral = (self.neutral / total) * 100
            self.volatile = (self.volatile / total) * 100
    
    def is_healthy(self) -> bool:
        """Check if spectrum indicates healthy stability distribution."""
        return (
            5 < self.protected < 25 and
            50 < self.neutral < 85 and
            10 < self.volatile < 30
        )
    
    def diagnosis(self) -> str:
        """Provide diagnosis of stability health."""
        if self.protected > 25:
            return "Over-protected: Model may be too rigid"
        elif self.protected < 5:
            return "Under-protected: Identity at risk"
        elif self.volatile > 30:
            return "Too volatile: Unstable learning"
        elif self.volatile < 10:
            return "Too stable: Limited learning capacity"
        else:
            return "Healthy: Balanced stability spectrum"


class StabilityAnalyzer:
    """
    Analyzes parameter stability across the model.
    
    Uses multiple signals:
    - Weight change magnitude
    - Gradient consistency
    - Update frequency
    - Value variance over time
    """
    
    def __init__(
        self,
        model: nn.Module,
        protected_threshold: float = 0.7,
        volatile_threshold: float = 0.3,
        history_length: int = 50,
    ):
        """
        Initialize StabilityAnalyzer.
        
        Args:
            model: The neural network to analyze
            protected_threshold: Score above this → protected
            volatile_threshold: Score below this → volatile
            history_length: Number of steps to track
        """
        self.model = model
        self.protected_threshold = protected_threshold
        self.volatile_threshold = volatile_threshold
        self.history_length = history_length
        
        # History tracking
        self.weight_history: Dict[str, List[torch.Tensor]] = {}
        self.gradient_history: Dict[str, List[torch.Tensor]] = {}
        self.stability_history: Dict[str, List[float]] = {}
        
        # Current state
        self.stability_scores: Dict[str, float] = {}
        self.stability_states: Dict[str, StabilityState] = {}
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize tracking for all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.weight_history[name] = []
                self.gradient_history[name] = []
                self.stability_history[name] = []
                self.stability_scores[name] = 0.5
                self.stability_states[name] = StabilityState.NEUTRAL
    
    def update(self) -> None:
        """Update history with current model state."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Track weights
            self.weight_history[name].append(param.data.clone().cpu())
            if len(self.weight_history[name]) > self.history_length:
                self.weight_history[name].pop(0)
            
            # Track gradients
            if param.grad is not None:
                self.gradient_history[name].append(param.grad.data.clone().cpu())
                if len(self.gradient_history[name]) > self.history_length:
                    self.gradient_history[name].pop(0)
    
    def analyze(self) -> Dict[str, float]:
        """
        Analyze stability of all parameters.
        
        Returns:
            Dictionary of parameter names to stability scores (0-1)
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            score = self._compute_stability(name)
            self.stability_scores[name] = score
            self.stability_states[name] = self._classify_state(score)
            
            # Track history
            self.stability_history[name].append(score)
            if len(self.stability_history[name]) > self.history_length:
                self.stability_history[name].pop(0)
        
        return self.stability_scores.copy()
    
    def _compute_stability(self, name: str) -> float:
        """
        Compute stability score for a parameter.
        
        Combines:
        - Weight variance (low variance = stable)
        - Gradient consistency (consistent direction = stable)
        - Change magnitude (small changes = stable)
        """
        scores = []
        
        # Weight variance score
        if len(self.weight_history[name]) >= 2:
            weights = torch.stack(self.weight_history[name])
            variance = weights.var(dim=0).mean().item()
            # Normalize: low variance = high stability
            weight_score = 1.0 / (1.0 + variance * 100)
            scores.append(weight_score)
        
        # Gradient consistency score
        if len(self.gradient_history[name]) >= 2:
            grads = self.gradient_history[name]
            consistencies = []
            for i in range(1, len(grads)):
                prev = grads[i-1].flatten()
                curr = grads[i].flatten()
                if torch.norm(prev) > 1e-8 and torch.norm(curr) > 1e-8:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        prev.unsqueeze(0), curr.unsqueeze(0)
                    ).item()
                    consistencies.append((cos_sim + 1) / 2)  # Map to 0-1
            
            if consistencies:
                grad_score = sum(consistencies) / len(consistencies)
                scores.append(grad_score)
        
        # Change magnitude score
        if len(self.weight_history[name]) >= 2:
            first = self.weight_history[name][0]
            last = self.weight_history[name][-1]
            change = torch.norm(last - first).item()
            # Normalize: small change = high stability
            change_score = 1.0 / (1.0 + change)
            scores.append(change_score)
        
        # Combine scores
        if not scores:
            return 0.5  # Default neutral
        
        return sum(scores) / len(scores)
    
    def _classify_state(self, score: float) -> StabilityState:
        """Classify score into stability state."""
        if score >= self.protected_threshold:
            return StabilityState.PROTECTED
        elif score <= self.volatile_threshold:
            return StabilityState.VOLATILE
        else:
            return StabilityState.NEUTRAL
    
    def classify(self) -> StabilitySpectrum:
        """
        Classify all parameters and return spectrum.
        
        Returns:
            StabilitySpectrum with percentage distribution
        """
        if not self.stability_states:
            self.analyze()
        
        total = len(self.stability_states)
        if total == 0:
            return StabilitySpectrum(0, 100, 0)
        
        protected = sum(
            1 for s in self.stability_states.values()
            if s == StabilityState.PROTECTED
        )
        volatile = sum(
            1 for s in self.stability_states.values()
            if s == StabilityState.VOLATILE
        )
        neutral = total - protected - volatile
        
        return StabilitySpectrum(
            protected=(protected / total) * 100,
            neutral=(neutral / total) * 100,
            volatile=(volatile / total) * 100,
        )
    
    def get_protected_params(self) -> List[str]:
        """Get names of all protected parameters."""
        return [
            name for name, state in self.stability_states.items()
            if state == StabilityState.PROTECTED
        ]
    
    def get_volatile_params(self) -> List[str]:
        """Get names of all volatile parameters."""
        return [
            name for name, state in self.stability_states.items()
            if state == StabilityState.VOLATILE
        ]


def protect_mask(
    model: nn.Module,
    stability_scores: Dict[str, float],
    threshold: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """
    Create protection masks for all parameters.
    
    Args:
        model: The neural network
        stability_scores: Stability score per parameter
        threshold: Protection threshold
        
    Returns:
        Dictionary of parameter names to protection masks (0-1)
    """
    masks = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        score = stability_scores.get(name, 0.5)
        
        if score >= threshold:
            # Protected: scale down updates
            protection_strength = (score - threshold) / (1.0 - threshold)
            mask = torch.ones_like(param.data) * (1.0 - protection_strength)
        else:
            # Not protected: full updates allowed
            mask = torch.ones_like(param.data)
        
        masks[name] = mask
    
    return masks


def drift_estimator(
    current_weights: Dict[str, torch.Tensor],
    reference_weights: Dict[str, torch.Tensor],
    normalize: bool = True,
) -> float:
    """
    Estimate semantic drift from reference state.
    
    Args:
        current_weights: Current model weights
        reference_weights: Reference (original) weights
        normalize: Whether to normalize by number of parameters
        
    Returns:
        Drift amount (0-1 if normalized)
    """
    total_drift = 0.0
    total_params = 0
    
    for name in current_weights:
        if name not in reference_weights:
            continue
        
        current = current_weights[name]
        reference = reference_weights[name]
        
        # L2 distance
        drift = torch.norm(current - reference).item()
        total_drift += drift
        total_params += current.numel()
    
    if normalize and total_params > 0:
        # Normalize to 0-1 range (approximate)
        return min(total_drift / (total_params ** 0.5), 1.0)
    
    return total_drift
