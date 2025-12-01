"""
SAL Pulse-Split-Cascade (PSC) Module

A semantic Game of Life for neural networks.
Patterns emerge, split, cascade, and the best lineages persist.

No rewards. No scores. Just resonance-based selection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
import uuid


class PulseState(Enum):
    """State of a pulse in the cascade."""
    ACTIVE = "active"       # Currently processing
    SPLIT = "split"         # Has spawned children
    MERGED = "merged"       # Has been merged into lineage
    DORMANT = "dormant"     # Inactive but preserved
    EXPIRED = "expired"     # No longer viable


@dataclass
class Pulse:
    """
    A single pulse in the semantic cascade.
    
    A pulse is a snapshot of semantic state that can:
    - Evolve independently
    - Split into multiple branches
    - Merge with compatible pulses
    - Expire if it loses coherence
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: PulseState = PulseState.ACTIVE
    generation: int = 0
    parent_id: Optional[str] = None
    
    # Semantic content
    embedding: Optional[torch.Tensor] = None
    coherence: float = 0.5
    novelty: float = 0.5
    resonance: float = 0.5
    
    # Lineage tracking
    children: List[str] = field(default_factory=list)
    birth_step: int = 0
    last_active_step: int = 0
    
    def fitness(self) -> float:
        """
        Compute fitness without reward.
        
        Fitness = coherence × (novelty + resonance) / 2
        
        This is NOT optimization — it's observation of natural quality.
        """
        return self.coherence * (self.novelty + self.resonance) / 2
    
    def is_viable(self) -> bool:
        """Check if pulse is still viable."""
        return (
            self.state == PulseState.ACTIVE and
            self.coherence > 0.3 and
            self.resonance > 0.2
        )
    
    def can_split(self) -> bool:
        """Check if pulse can split into children."""
        return (
            self.is_viable() and
            self.coherence > 0.5 and
            self.novelty > 0.3
        )


@dataclass
class Lineage:
    """
    A lineage is a family of related pulses.
    
    Lineages track the evolution of semantic patterns
    through the cascade, preserving successful structures.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    root_pulse_id: str = ""
    
    # All pulses in this lineage
    pulses: Dict[str, Pulse] = field(default_factory=dict)
    
    # Best pulse in lineage
    best_pulse_id: Optional[str] = None
    best_fitness: float = 0.0
    
    # Lineage statistics
    total_generations: int = 0
    active_pulses: int = 0
    merged_count: int = 0
    
    def add_pulse(self, pulse: Pulse) -> None:
        """Add a pulse to this lineage."""
        self.pulses[pulse.id] = pulse
        
        # Update statistics
        self.total_generations = max(self.total_generations, pulse.generation + 1)
        if pulse.state == PulseState.ACTIVE:
            self.active_pulses += 1
        
        # Track best
        fitness = pulse.fitness()
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_pulse_id = pulse.id
    
    def get_active_pulses(self) -> List[Pulse]:
        """Get all active pulses in lineage."""
        return [p for p in self.pulses.values() if p.state == PulseState.ACTIVE]
    
    def get_best_pulse(self) -> Optional[Pulse]:
        """Get the best pulse in lineage."""
        if self.best_pulse_id:
            return self.pulses.get(self.best_pulse_id)
        return None
    
    def overall_fitness(self) -> float:
        """Compute overall lineage fitness."""
        if not self.pulses:
            return 0.0
        
        active = self.get_active_pulses()
        if not active:
            # Use best historical
            return self.best_fitness * 0.8  # Decay for inactive
        
        # Average of active pulses
        return sum(p.fitness() for p in active) / len(active)


class PulseCascade:
    """
    The full Pulse-Split-Cascade system.
    
    This is semantic Game of Life:
    1. Prompt creates initial pulse
    2. Pulse splits into branches
    3. Branches evolve independently
    4. Compatible branches merge into lineages
    5. Best lineage emerges naturally
    
    No optimization. No rewards. Just emergence.
    """
    
    def __init__(
        self,
        max_pulses: int = 32,
        max_generations: int = 10,
        split_threshold: float = 0.6,
        merge_threshold: float = 0.8,
        expire_threshold: float = 0.3,
    ):
        """
        Initialize PulseCascade.
        
        Args:
            max_pulses: Maximum concurrent pulses
            max_generations: Maximum pulse generations
            split_threshold: Coherence needed to split
            merge_threshold: Similarity needed to merge
            expire_threshold: Minimum coherence to survive
        """
        self.max_pulses = max_pulses
        self.max_generations = max_generations
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.expire_threshold = expire_threshold
        
        # Active state
        self.pulses: Dict[str, Pulse] = {}
        self.lineages: Dict[str, Lineage] = {}
        
        # Tracking
        self.current_step = 0
        self.total_pulses_created = 0
        self.total_merges = 0
        self.emergence_events: List[Dict] = []
    
    def initiate(self, embedding: torch.Tensor) -> Pulse:
        """
        Initiate cascade from a prompt embedding.
        
        Args:
            embedding: Initial semantic embedding
            
        Returns:
            Root pulse of the cascade
        """
        pulse = Pulse(
            embedding=embedding.clone(),
            generation=0,
            birth_step=self.current_step,
            last_active_step=self.current_step,
            coherence=1.0,  # Initial pulse is maximally coherent
            novelty=1.0,    # Initial pulse is maximally novel
            resonance=0.5,  # Neutral resonance initially
        )
        
        self.pulses[pulse.id] = pulse
        self.total_pulses_created += 1
        
        # Create root lineage
        lineage = Lineage(root_pulse_id=pulse.id)
        lineage.add_pulse(pulse)
        self.lineages[lineage.id] = lineage
        
        return pulse
    
    def step(
        self,
        evolve_fn: Callable[[torch.Tensor], torch.Tensor],
        measure_fn: Optional[Callable[[torch.Tensor], Tuple[float, float, float]]] = None,
    ) -> List[Pulse]:
        """
        Advance the cascade by one step.
        
        Args:
            evolve_fn: Function to evolve embeddings
            measure_fn: Optional function to measure (coherence, novelty, resonance)
            
        Returns:
            List of currently active pulses
        """
        self.current_step += 1
        
        active_pulses = [p for p in self.pulses.values() if p.is_viable()]
        new_pulses = []
        
        for pulse in active_pulses:
            # Evolve
            if pulse.embedding is not None:
                evolved = evolve_fn(pulse.embedding)
                pulse.embedding = evolved
                pulse.last_active_step = self.current_step
                
                # Measure
                if measure_fn:
                    c, n, r = measure_fn(evolved)
                    pulse.coherence = c
                    pulse.novelty = n
                    pulse.resonance = r
            
            # Check for split
            if (
                pulse.can_split() and
                pulse.generation < self.max_generations and
                len(self.pulses) < self.max_pulses
            ):
                children = self._split_pulse(pulse, evolve_fn)
                new_pulses.extend(children)
            
            # Check for expiration
            if pulse.coherence < self.expire_threshold:
                pulse.state = PulseState.EXPIRED
        
        # Add new pulses
        for p in new_pulses:
            self.pulses[p.id] = p
        
        # Attempt merges
        self._attempt_merges()
        
        return [p for p in self.pulses.values() if p.is_viable()]
    
    def _split_pulse(
        self,
        pulse: Pulse,
        evolve_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[Pulse]:
        """Split a pulse into children."""
        if pulse.embedding is None:
            return []
        
        children = []
        num_children = 2  # Binary split
        
        for i in range(num_children):
            # Add variation
            noise = torch.randn_like(pulse.embedding) * 0.1
            child_embedding = pulse.embedding + noise
            
            child = Pulse(
                embedding=child_embedding,
                generation=pulse.generation + 1,
                parent_id=pulse.id,
                birth_step=self.current_step,
                last_active_step=self.current_step,
                coherence=pulse.coherence * 0.9,  # Slight degradation
                novelty=min(pulse.novelty + 0.1, 1.0),  # Increase novelty
                resonance=pulse.resonance,
            )
            
            children.append(child)
            pulse.children.append(child.id)
            self.total_pulses_created += 1
        
        pulse.state = PulseState.SPLIT
        return children
    
    def _attempt_merges(self) -> None:
        """Attempt to merge compatible pulses."""
        active = [p for p in self.pulses.values() if p.is_viable()]
        
        merged = set()
        
        for i, p1 in enumerate(active):
            if p1.id in merged:
                continue
            
            for p2 in active[i+1:]:
                if p2.id in merged:
                    continue
                
                if p1.embedding is None or p2.embedding is None:
                    continue
                
                # Check similarity
                sim = torch.nn.functional.cosine_similarity(
                    p1.embedding.flatten().unsqueeze(0),
                    p2.embedding.flatten().unsqueeze(0)
                ).item()
                
                if sim > self.merge_threshold:
                    # Merge: combine into p1
                    p1.embedding = (p1.embedding + p2.embedding) / 2
                    p1.coherence = max(p1.coherence, p2.coherence)
                    p1.novelty = (p1.novelty + p2.novelty) / 2
                    p1.resonance = max(p1.resonance, p2.resonance)
                    
                    p2.state = PulseState.MERGED
                    merged.add(p2.id)
                    self.total_merges += 1
    
    def emerge(self) -> Optional[Pulse]:
        """
        Get the emergent pulse — the best that has naturally arisen.
        
        This is NOT selection by score. This is observation of what emerged.
        
        Returns:
            The most coherent, resonant pulse, or None
        """
        viable = [p for p in self.pulses.values() if p.is_viable()]
        
        if not viable:
            # Fall back to best historical
            all_pulses = list(self.pulses.values())
            if not all_pulses:
                return None
            return max(all_pulses, key=lambda p: p.fitness())
        
        # Natural emergence: highest fitness among viable
        emergent = max(viable, key=lambda p: p.fitness())
        
        # Record emergence event
        self.emergence_events.append({
            'step': self.current_step,
            'pulse_id': emergent.id,
            'generation': emergent.generation,
            'coherence': emergent.coherence,
            'novelty': emergent.novelty,
            'resonance': emergent.resonance,
            'fitness': emergent.fitness(),
        })
        
        return emergent
    
    def get_best_lineage(self) -> Optional[Lineage]:
        """Get the best performing lineage."""
        if not self.lineages:
            return None
        return max(self.lineages.values(), key=lambda l: l.overall_fitness())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cascade statistics."""
        active = sum(1 for p in self.pulses.values() if p.is_viable())
        expired = sum(1 for p in self.pulses.values() if p.state == PulseState.EXPIRED)
        merged = sum(1 for p in self.pulses.values() if p.state == PulseState.MERGED)
        
        return {
            'current_step': self.current_step,
            'total_pulses_created': self.total_pulses_created,
            'active_pulses': active,
            'expired_pulses': expired,
            'merged_pulses': merged,
            'total_merges': self.total_merges,
            'num_lineages': len(self.lineages),
            'emergence_events': len(self.emergence_events),
        }


def emergence_select(pulses: List[Pulse]) -> Optional[Pulse]:
    """
    Select the emergent pulse from a list.
    
    This is NOT optimization. This is observing which pattern
    has naturally become the most coherent and resonant.
    
    Args:
        pulses: List of pulses to select from
        
    Returns:
        The emergent pulse, or None
    """
    if not pulses:
        return None
    
    # Filter to viable only
    viable = [p for p in pulses if p.is_viable()]
    
    if not viable:
        # Fall back to best fitness among all
        return max(pulses, key=lambda p: p.fitness())
    
    # Natural emergence
    return max(viable, key=lambda p: p.fitness())
