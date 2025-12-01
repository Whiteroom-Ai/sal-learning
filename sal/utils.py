"""
SAL Utilities Module

Helper functions for SAL operations.
Similarity measures, smoothing, seed loading.
"""

import torch
import json
from typing import Dict, Optional, Any, List, Union
from pathlib import Path


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute cosine similarity between tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small epsilon for numerical stability
        
    Returns:
        Cosine similarity (same shape as input, minus the compared dimension)
    """
    a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
    
    return (a_norm * b_norm).sum(dim=dim)


def exponential_moving_average(
    current: torch.Tensor,
    previous: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Compute exponential moving average.
    
    EMA = alpha * current + (1 - alpha) * previous
    
    Args:
        current: Current value
        previous: Previous EMA value
        alpha: Smoothing factor (0-1, higher = more weight on current)
        
    Returns:
        Updated EMA
    """
    return alpha * current + (1 - alpha) * previous


class EMA:
    """
    Exponential Moving Average tracker.
    
    Useful for smoothing stability scores and other metrics.
    """
    
    def __init__(self, alpha: float = 0.1, initial: Optional[float] = None):
        """
        Initialize EMA tracker.
        
        Args:
            alpha: Smoothing factor
            initial: Initial value (None = use first update)
        """
        self.alpha = alpha
        self.value = initial
        self.count = 0
    
    def update(self, new_value: float) -> float:
        """
        Update EMA with new value.
        
        Args:
            new_value: New observation
            
        Returns:
            Updated EMA value
        """
        self.count += 1
        
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def get(self) -> Optional[float]:
        """Get current EMA value."""
        return self.value
    
    def reset(self) -> None:
        """Reset EMA tracker."""
        self.value = None
        self.count = 0


def load_seed(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a semantic seed from JSON file.
    
    Seeds are anchor points in semantic space that help
    maintain identity and coherence.
    
    Args:
        path: Path to seed JSON file
        
    Returns:
        Seed dictionary with embedding and metadata
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        seed = json.load(f)
    
    # Convert embedding to tensor if present
    if 'embedding' in seed and isinstance(seed['embedding'], list):
        seed['embedding'] = torch.tensor(seed['embedding'])
    
    return seed


def save_seed(
    seed: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save a semantic seed to JSON file.
    
    Args:
        seed: Seed dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to list for JSON
    seed_copy = seed.copy()
    if 'embedding' in seed_copy and isinstance(seed_copy['embedding'], torch.Tensor):
        seed_copy['embedding'] = seed_copy['embedding'].tolist()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(seed_copy, f, indent=2, ensure_ascii=False)


def create_seed(
    name: str,
    dimension: int = 768,
    seed_type: str = "random",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create a new semantic seed.
    
    Args:
        name: Seed name
        dimension: Embedding dimension
        seed_type: Type of seed initialization
        metadata: Additional metadata
        
    Returns:
        Seed dictionary
    """
    if seed_type == "random":
        embedding = torch.randn(dimension)
        embedding = embedding / embedding.norm()  # Normalize
    elif seed_type == "zero":
        embedding = torch.zeros(dimension)
    elif seed_type == "ones":
        embedding = torch.ones(dimension) / (dimension ** 0.5)
    else:
        raise ValueError(f"Unknown seed type: {seed_type}")
    
    seed = {
        'name': name,
        'dimension': dimension,
        'type': seed_type,
        'embedding': embedding,
        'metadata': metadata or {},
    }
    
    return seed


def weight_distance(
    weights1: Dict[str, torch.Tensor],
    weights2: Dict[str, torch.Tensor],
    metric: str = "l2",
) -> float:
    """
    Compute distance between two sets of weights.
    
    Args:
        weights1: First weight dictionary
        weights2: Second weight dictionary
        metric: Distance metric ("l2", "l1", "cosine")
        
    Returns:
        Distance value
    """
    total_distance = 0.0
    count = 0
    
    for name in weights1:
        if name not in weights2:
            continue
        
        w1 = weights1[name].flatten().float()
        w2 = weights2[name].flatten().float()
        
        if w1.shape != w2.shape:
            continue
        
        if metric == "l2":
            dist = torch.norm(w1 - w2).item()
        elif metric == "l1":
            dist = torch.abs(w1 - w2).sum().item()
        elif metric == "cosine":
            cos_sim = cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
            dist = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        total_distance += dist
        count += 1
    
    if count == 0:
        return 0.0
    
    return total_distance / count


def gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm across model.
    
    Args:
        model: Neural network
        
    Returns:
        Total gradient L2 norm
    """
    total_norm = 0.0
    
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    
    return total_norm ** 0.5


def parameter_count(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """
    Count parameters in model.
    
    Args:
        model: Neural network
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def stability_summary(stability_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Summarize stability scores.
    
    Args:
        stability_scores: Dictionary of parameter names to scores
        
    Returns:
        Summary with mean, std, min, max, and distribution
    """
    if not stability_scores:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'protected_pct': 0.0,
            'neutral_pct': 0.0,
            'volatile_pct': 0.0,
        }
    
    scores = list(stability_scores.values())
    n = len(scores)
    
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5
    
    protected = sum(1 for s in scores if s > 0.7) / n * 100
    volatile = sum(1 for s in scores if s < 0.3) / n * 100
    neutral = 100 - protected - volatile
    
    return {
        'mean': mean,
        'std': std,
        'min': min(scores),
        'max': max(scores),
        'protected_pct': protected,
        'neutral_pct': neutral,
        'volatile_pct': volatile,
    }
