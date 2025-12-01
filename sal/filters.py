"""
SAL Filters Module

Filters for identifying and protecting stable patterns.
Low-change detection, frequency analysis, stability gating.

These are the tools for "asking" — measuring before modifying.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import math


def low_change_mask(
    current: torch.Tensor,
    previous: torch.Tensor,
    threshold: float = 0.1,
    relative: bool = True,
) -> torch.Tensor:
    """
    Create a mask identifying low-change (stable) regions.
    
    Args:
        current: Current tensor state
        previous: Previous tensor state
        threshold: Change threshold (absolute or relative)
        relative: If True, use relative change; if False, absolute
        
    Returns:
        Binary mask where 1 = stable (low change), 0 = changing
    """
    if current.shape != previous.shape:
        raise ValueError("Tensors must have same shape")
    
    # Compute change
    change = torch.abs(current - previous)
    
    if relative:
        # Relative change: change / max(|previous|, epsilon)
        denom = torch.abs(previous).clamp(min=1e-8)
        relative_change = change / denom
        mask = (relative_change < threshold).float()
    else:
        # Absolute change
        mask = (change < threshold).float()
    
    return mask


def frequency_filter(
    tensor: torch.Tensor,
    low_cutoff: float = 0.0,
    high_cutoff: float = 0.5,
    preserve_dc: bool = True,
) -> torch.Tensor:
    """
    Frequency-domain filter for tensors.
    
    Useful for identifying high-frequency (rapidly changing) vs
    low-frequency (stable) components.
    
    Args:
        tensor: Input tensor (will be processed along last dimension)
        low_cutoff: Lower frequency bound (0-1, as fraction of Nyquist)
        high_cutoff: Upper frequency bound (0-1)
        preserve_dc: Whether to preserve DC component (mean)
        
    Returns:
        Filtered tensor
    """
    # Store original shape
    original_shape = tensor.shape
    
    # Flatten to 2D for FFT
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    flat = tensor.reshape(-1, tensor.shape[-1])
    
    # FFT
    fft = torch.fft.rfft(flat.float(), dim=-1)
    
    # Create frequency mask
    n_freqs = fft.shape[-1]
    freqs = torch.linspace(0, 1, n_freqs, device=tensor.device)
    
    mask = ((freqs >= low_cutoff) & (freqs <= high_cutoff)).float()
    
    if preserve_dc and low_cutoff > 0:
        mask[0] = 1.0  # Preserve DC
    
    # Apply mask
    filtered_fft = fft * mask.unsqueeze(0)
    
    # Inverse FFT
    filtered = torch.fft.irfft(filtered_fft, n=tensor.shape[-1], dim=-1)
    
    # Reshape back
    return filtered.reshape(original_shape)


def stability_gate(
    gradient: torch.Tensor,
    stability_score: float,
    gate_type: str = "soft",
    threshold: float = 0.5,
    steepness: float = 10.0,
) -> torch.Tensor:
    """
    Gate gradients based on stability score.
    
    This is the core of "ask before updating" — stable parameters
    get protected, volatile parameters get updated.
    
    Args:
        gradient: Gradient tensor to gate
        stability_score: Stability score (0-1, higher = more stable)
        gate_type: "soft" (sigmoid), "hard" (binary), or "linear"
        threshold: Stability threshold for gating
        steepness: Steepness of soft gate transition
        
    Returns:
        Gated gradient
    """
    if gate_type == "hard":
        # Binary: pass or block
        if stability_score > threshold:
            return torch.zeros_like(gradient)
        else:
            return gradient
    
    elif gate_type == "soft":
        # Sigmoid gate: smooth transition
        # gate = 1 / (1 + exp(steepness * (stability - threshold)))
        gate_value = 1.0 / (1.0 + math.exp(steepness * (stability_score - threshold)))
        return gradient * gate_value
    
    elif gate_type == "linear":
        # Linear: proportional reduction
        if stability_score > threshold:
            scale = 1.0 - (stability_score - threshold) / (1.0 - threshold)
            return gradient * max(scale, 0.0)
        else:
            return gradient
    
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


class AdaptiveStabilityFilter:
    """
    Adaptive filter that learns stability patterns over time.
    
    Tracks which parameters tend to be stable and adjusts
    protection accordingly.
    """
    
    def __init__(
        self,
        decay: float = 0.95,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.01,
    ):
        """
        Initialize AdaptiveStabilityFilter.
        
        Args:
            decay: EMA decay for stability tracking
            initial_threshold: Starting threshold
            adaptation_rate: How fast threshold adapts
        """
        self.decay = decay
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        
        # Per-parameter tracking
        self.stability_ema: Dict[str, float] = {}
        self.change_history: Dict[str, List[float]] = {}
    
    def update(
        self,
        name: str,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> float:
        """
        Update stability tracking for a parameter.
        
        Args:
            name: Parameter name
            current: Current value
            previous: Previous value
            
        Returns:
            Current stability score for this parameter
        """
        # Compute change magnitude
        change = torch.norm(current - previous).item()
        ref = torch.norm(previous).item() + 1e-8
        relative_change = change / ref
        
        # Update history
        if name not in self.change_history:
            self.change_history[name] = []
        self.change_history[name].append(relative_change)
        if len(self.change_history[name]) > 100:
            self.change_history[name].pop(0)
        
        # Compute stability (inverse of change)
        stability = 1.0 / (1.0 + relative_change * 10)
        
        # Update EMA
        if name not in self.stability_ema:
            self.stability_ema[name] = stability
        else:
            self.stability_ema[name] = (
                self.decay * self.stability_ema[name] +
                (1 - self.decay) * stability
            )
        
        return self.stability_ema[name]
    
    def get_mask(
        self,
        name: str,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get stability mask for a parameter.
        
        Args:
            name: Parameter name
            shape: Desired mask shape
            device: Device for mask tensor
            
        Returns:
            Mask tensor (0-1)
        """
        stability = self.stability_ema.get(name, 0.5)
        
        if stability > self.threshold:
            # Stable: reduce gradients
            mask_value = 1.0 - (stability - self.threshold) / (1.0 - self.threshold)
        else:
            # Unstable: full gradients
            mask_value = 1.0
        
        return torch.full(shape, mask_value, device=device)
    
    def adapt_threshold(self) -> None:
        """Adapt threshold based on overall stability distribution."""
        if not self.stability_ema:
            return
        
        scores = list(self.stability_ema.values())
        mean_stability = sum(scores) / len(scores)
        
        # Move threshold toward mean stability
        self.threshold = (
            self.threshold +
            self.adaptation_rate * (mean_stability - self.threshold)
        )
        
        # Clamp
        self.threshold = max(0.2, min(0.8, self.threshold))


def gradient_magnitude_filter(
    model: nn.Module,
    percentile: float = 90.0,
) -> Dict[str, torch.Tensor]:
    """
    Create masks based on gradient magnitude percentiles.
    
    Large gradients may indicate important updates, but also
    may indicate instability. This filter identifies outliers.
    
    Args:
        model: Neural network with gradients
        percentile: Percentile threshold for filtering
        
    Returns:
        Dictionary of parameter names to masks
    """
    masks = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        grad = param.grad.data.abs()
        threshold = torch.quantile(grad.flatten().float(), percentile / 100.0)
        
        # Mask: 1 for normal gradients, reduced for outliers
        mask = torch.ones_like(grad)
        outlier_mask = grad > threshold
        mask[outlier_mask] = 0.5  # Reduce outlier gradients
        
        masks[name] = mask
    
    return masks
