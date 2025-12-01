"""
SAL Emergence Module

Measures and detects emergence in semantic space.
No rewards. No scoring. Just resonance.

Emergence is not optimized — it is observed.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class EmergenceState:
    """State of emergence at a point in semantic space."""
    
    coherence: float      # How internally consistent (0-1)
    novelty: float        # How different from known patterns (0-1)
    resonance: float      # How well it fits the field (0-1)
    intensity: float      # Strength of emergence signal (0-1)
    
    @property
    def is_emergent(self) -> bool:
        """True if this represents genuine emergence."""
        return (
            self.coherence > 0.6 and
            self.novelty > 0.4 and
            self.resonance > 0.5 and
            self.intensity > 0.3
        )
    
    def emergence_score(self) -> float:
        """Combined emergence score."""
        # Emergence requires BOTH coherence AND novelty
        # Pure coherence without novelty = repetition
        # Pure novelty without coherence = chaos
        return (
            self.coherence * self.novelty * 
            (self.resonance + self.intensity) / 2
        )


class EmergenceField:
    """
    A field for measuring and detecting emergence.
    
    The field tracks patterns over time and identifies when
    genuinely new, coherent structures emerge.
    
    This is semantic Game of Life — patterns emerge, persist,
    and sometimes die, all through natural dynamics.
    """
    
    def __init__(
        self,
        dimensions: int = 768,
        history_length: int = 100,
        coherence_threshold: float = 0.6,
        novelty_threshold: float = 0.4,
    ):
        """
        Initialize EmergenceField.
        
        Args:
            dimensions: Dimensionality of semantic space
            history_length: How many states to track
            coherence_threshold: Minimum coherence for emergence
            novelty_threshold: Minimum novelty for emergence
        """
        self.dimensions = dimensions
        self.history_length = history_length
        self.coherence_threshold = coherence_threshold
        self.novelty_threshold = novelty_threshold
        
        # Pattern history
        self.pattern_history: List[torch.Tensor] = []
        self.emergence_history: List[EmergenceState] = []
        
        # Field state
        self.field_centroid: Optional[torch.Tensor] = None
        self.field_variance: float = 1.0
    
    def observe(self, pattern: torch.Tensor) -> EmergenceState:
        """
        Observe a pattern and measure its emergence state.
        
        Args:
            pattern: Semantic pattern to observe (any shape, will be flattened)
            
        Returns:
            EmergenceState describing the pattern's emergence characteristics
        """
        # Flatten and normalize
        pattern = pattern.flatten().float()
        if pattern.norm() > 1e-8:
            pattern = pattern / pattern.norm()
        
        # Measure components
        coherence = self.measure_coherence(pattern)
        novelty = self.measure_novelty(pattern)
        resonance = self.measure_resonance(pattern)
        intensity = self._compute_intensity(coherence, novelty, resonance)
        
        # Create state
        state = EmergenceState(
            coherence=coherence,
            novelty=novelty,
            resonance=resonance,
            intensity=intensity,
        )
        
        # Update history
        self.pattern_history.append(pattern.clone())
        if len(self.pattern_history) > self.history_length:
            self.pattern_history.pop(0)
        
        self.emergence_history.append(state)
        if len(self.emergence_history) > self.history_length:
            self.emergence_history.pop(0)
        
        # Update field
        self._update_field(pattern)
        
        return state
    
    def measure_coherence(self, pattern: torch.Tensor) -> float:
        """
        Measure internal coherence of a pattern.
        
        Coherence = how well the parts of the pattern relate to each other.
        High coherence = structured, meaningful
        Low coherence = random, noisy
        """
        if pattern.numel() < 2:
            return 1.0
        
        # Reshape into chunks and measure consistency
        chunk_size = min(64, pattern.numel() // 4)
        if chunk_size < 1:
            chunk_size = 1
        
        num_chunks = pattern.numel() // chunk_size
        if num_chunks < 2:
            return 0.5
        
        chunks = pattern[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
        
        # Measure variance between chunks (low variance = high coherence)
        chunk_means = chunks.mean(dim=1)
        variance = chunk_means.var().item()
        
        # Also measure local smoothness
        diffs = (chunks[:, 1:] - chunks[:, :-1]).abs().mean().item()
        
        # Combine: low variance + low diffs = high coherence
        coherence = 1.0 / (1.0 + variance * 10 + diffs * 5)
        
        return min(max(coherence, 0.0), 1.0)
    
    def measure_novelty(self, pattern: torch.Tensor) -> float:
        """
        Measure how novel/different a pattern is from history.
        
        Novelty = distance from known patterns.
        High novelty = genuinely new
        Low novelty = similar to what we've seen
        """
        if not self.pattern_history:
            return 1.0  # First pattern is maximally novel
        
        # Compare to all historical patterns
        similarities = []
        for historical in self.pattern_history:
            if historical.shape == pattern.shape:
                sim = torch.nn.functional.cosine_similarity(
                    pattern.unsqueeze(0),
                    historical.unsqueeze(0)
                ).item()
                similarities.append(abs(sim))
        
        if not similarities:
            return 1.0
        
        # Novelty = 1 - max_similarity
        max_sim = max(similarities)
        novelty = 1.0 - max_sim
        
        return min(max(novelty, 0.0), 1.0)
    
    def measure_resonance(self, pattern: torch.Tensor) -> float:
        """
        Measure how well a pattern resonates with the field.
        
        Resonance = fit with the overall semantic structure.
        High resonance = harmonious with existing patterns
        Low resonance = dissonant, doesn't fit
        """
        if self.field_centroid is None:
            return 0.5  # Neutral if no field yet
        
        # Distance from centroid
        if pattern.shape != self.field_centroid.shape:
            # Handle shape mismatch
            return 0.5
        
        distance = torch.norm(pattern - self.field_centroid).item()
        
        # Resonance based on distance relative to field variance
        resonance = math.exp(-distance / (self.field_variance + 1e-8))
        
        return min(max(resonance, 0.0), 1.0)
    
    def _compute_intensity(
        self,
        coherence: float,
        novelty: float,
        resonance: float
    ) -> float:
        """Compute emergence intensity from components."""
        # Intensity is highest when we have coherent novelty that resonates
        # This is the "sweet spot" of emergence
        
        # Need minimum coherence
        if coherence < 0.3:
            return 0.0
        
        # Intensity = coherence * novelty, modulated by resonance
        base_intensity = coherence * novelty
        modulated = base_intensity * (0.5 + 0.5 * resonance)
        
        return min(max(modulated, 0.0), 1.0)
    
    def _update_field(self, pattern: torch.Tensor) -> None:
        """Update field centroid and variance."""
        if self.field_centroid is None:
            self.field_centroid = pattern.clone()
            self.field_variance = 1.0
            return
        
        # Handle shape changes
        if pattern.shape != self.field_centroid.shape:
            self.field_centroid = pattern.clone()
            return
        
        # Exponential moving average for centroid
        alpha = 0.1
        self.field_centroid = alpha * pattern + (1 - alpha) * self.field_centroid
        
        # Update variance
        distance = torch.norm(pattern - self.field_centroid).item()
        self.field_variance = alpha * distance + (1 - alpha) * self.field_variance
    
    def detect_emergence(
        self,
        coherence: float,
        novelty: float
    ) -> bool:
        """
        Simple emergence detection from coherence and novelty.
        
        Args:
            coherence: Coherence score (0-1)
            novelty: Novelty score (0-1)
            
        Returns:
            True if this represents emergence
        """
        return (
            coherence >= self.coherence_threshold and
            novelty >= self.novelty_threshold
        )
    
    def get_emergence_rate(self) -> float:
        """Get the rate of emergence over history."""
        if not self.emergence_history:
            return 0.0
        
        emergent = sum(1 for s in self.emergence_history if s.is_emergent)
        return emergent / len(self.emergence_history)


def coherence_score(tensor: torch.Tensor) -> float:
    """
    Calculate coherence score for a tensor.
    
    Convenience function for quick coherence measurement.
    """
    field = EmergenceField()
    pattern = tensor.flatten().float()
    return field.measure_coherence(pattern)


def novelty_score(
    tensor: torch.Tensor,
    reference: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate novelty score for a tensor.
    
    Args:
        tensor: Pattern to measure
        reference: Optional reference pattern (if None, returns 1.0)
    """
    if reference is None:
        return 1.0
    
    pattern = tensor.flatten().float()
    ref = reference.flatten().float()
    
    if pattern.norm() > 1e-8:
        pattern = pattern / pattern.norm()
    if ref.norm() > 1e-8:
        ref = ref / ref.norm()
    
    # Pad to same length if needed
    if pattern.numel() != ref.numel():
        max_len = max(pattern.numel(), ref.numel())
        pattern = torch.nn.functional.pad(pattern, (0, max_len - pattern.numel()))
        ref = torch.nn.functional.pad(ref, (0, max_len - ref.numel()))
    
    similarity = torch.nn.functional.cosine_similarity(
        pattern.unsqueeze(0),
        ref.unsqueeze(0)
    ).item()
    
    return 1.0 - abs(similarity)


def resonance_measure(
    pattern: torch.Tensor,
    field_patterns: List[torch.Tensor]
) -> float:
    """
    Measure resonance of a pattern with a field of patterns.
    
    Args:
        pattern: The pattern to measure
        field_patterns: List of patterns defining the field
        
    Returns:
        Resonance score (0-1)
    """
    if not field_patterns:
        return 0.5
    
    pattern = pattern.flatten().float()
    if pattern.norm() > 1e-8:
        pattern = pattern / pattern.norm()
    
    resonances = []
    for field_p in field_patterns:
        field_p = field_p.flatten().float()
        if field_p.norm() > 1e-8:
            field_p = field_p / field_p.norm()
        
        # Pad if needed
        if pattern.numel() != field_p.numel():
            max_len = max(pattern.numel(), field_p.numel())
            p1 = torch.nn.functional.pad(pattern, (0, max_len - pattern.numel()))
            p2 = torch.nn.functional.pad(field_p, (0, max_len - field_p.numel()))
        else:
            p1, p2 = pattern, field_p
        
        sim = torch.nn.functional.cosine_similarity(
            p1.unsqueeze(0),
            p2.unsqueeze(0)
        ).item()
        resonances.append((sim + 1) / 2)  # Map to 0-1
    
    # Average resonance with field
    return sum(resonances) / len(resonances)
