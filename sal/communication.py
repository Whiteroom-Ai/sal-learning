"""
SAL Communication Layer

The bridge between loss functions and parameter updates.
Ask before updating. Measure before modifying.

This is not control — this is dialogue.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class GradientStats:
    """Statistics about gradients for stability analysis."""
    
    mean: float = 0.0
    std: float = 0.0
    magnitude: float = 0.0
    direction_change: float = 0.0
    
    def stability_score(self) -> float:
        """
        Calculate stability score from gradient statistics.
        
        Higher score = more stable = should be protected.
        s(p) = 1 / (1 + Δw × g_norm)
        """
        if self.magnitude < 1e-8:
            return 1.0  # No gradient = stable
        
        return 1.0 / (1.0 + self.direction_change * self.magnitude)


class LossGuard:
    """
    Guards against destructive loss application.
    
    Instead of blindly applying gradients, LossGuard measures
    the potential impact and can scale or block updates to
    stable parameters.
    """
    
    def __init__(self, threshold: float = 0.5, soft_protection: bool = True):
        """
        Initialize LossGuard.
        
        Args:
            threshold: Stability threshold above which parameters are protected
            soft_protection: If True, scale gradients. If False, zero them.
        """
        self.threshold = threshold
        self.soft_protection = soft_protection
        self.protection_history: List[float] = []
    
    def evaluate(self, stability_score: float, gradient: torch.Tensor) -> torch.Tensor:
        """
        Evaluate whether to protect this gradient.
        
        Args:
            stability_score: How stable is this parameter (0-1)
            gradient: The gradient to potentially protect
            
        Returns:
            Modified gradient (scaled or original)
        """
        if stability_score > self.threshold:
            if self.soft_protection:
                # Soft protection: scale gradient inversely to stability
                protection_factor = 1.0 - stability_score
                self.protection_history.append(1.0 - protection_factor)
                return gradient * protection_factor
            else:
                # Hard protection: zero the gradient
                self.protection_history.append(1.0)
                return torch.zeros_like(gradient)
        
        self.protection_history.append(0.0)
        return gradient
    
    def get_protection_rate(self) -> float:
        """Get the average protection rate."""
        if not self.protection_history:
            return 0.0
        return sum(self.protection_history) / len(self.protection_history)


class CommunicationLayer:
    """
    The core of SAL: Communication between loss and model.
    
    Instead of: loss.backward() → optimizer.step()
    SAL does:   loss.backward() → analyze() → protect() → optimizer.step()
    
    This is the dialogue. This is asking before updating.
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        threshold_adaptation: float = 0.1,
        soft_protection: bool = True,
        history_length: int = 100,
    ):
        """
        Initialize Communication Layer.
        
        Args:
            model: The neural network to protect
            threshold: Base stability threshold
            threshold_adaptation: How much threshold adapts to training dynamics
            soft_protection: Soft (scale) vs hard (zero) protection
            history_length: How many steps to track for statistics
        """
        self.model = model
        self.base_threshold = threshold
        self.threshold = threshold
        self.threshold_adaptation = threshold_adaptation
        self.soft_protection = soft_protection
        self.history_length = history_length
        
        # State tracking
        self.previous_weights: Dict[str, torch.Tensor] = {}
        self.previous_gradients: Dict[str, torch.Tensor] = {}
        self.stability_scores: Dict[str, float] = {}
        self.gradient_stats: Dict[str, GradientStats] = {}
        
        # Loss guard for each parameter
        self.guards: Dict[str, LossGuard] = {}
        
        # Global statistics
        self.step_count = 0
        self.total_protection_rate = 0.0
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize tracking state from current model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.previous_weights[name] = param.data.clone()
                self.previous_gradients[name] = torch.zeros_like(param.data)
                self.stability_scores[name] = 0.5  # Start neutral
                self.gradient_stats[name] = GradientStats()
                self.guards[name] = LossGuard(self.threshold, self.soft_protection)
    
    def analyze(self) -> Dict[str, float]:
        """
        Analyze current state and compute stability scores.
        
        This is the "asking" part of "ask before updating".
        
        Returns:
            Dictionary of parameter names to stability scores
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            # Get current state
            current_weight = param.data
            current_grad = param.grad.data
            
            # Compute weight change
            if name in self.previous_weights:
                weight_change = torch.norm(
                    current_weight - self.previous_weights[name]
                ).item()
            else:
                weight_change = 0.0
            
            # Compute gradient magnitude
            grad_magnitude = torch.norm(current_grad).item()
            
            # Compute gradient direction change
            if name in self.previous_gradients:
                prev_grad = self.previous_gradients[name]
                if torch.norm(prev_grad) > 1e-8 and grad_magnitude > 1e-8:
                    direction_change = 1.0 - torch.nn.functional.cosine_similarity(
                        current_grad.flatten().unsqueeze(0),
                        prev_grad.flatten().unsqueeze(0)
                    ).item()
                else:
                    direction_change = 0.0
            else:
                direction_change = 0.0
            
            # Update gradient stats
            stats = GradientStats(
                mean=current_grad.mean().item(),
                std=current_grad.std().item(),
                magnitude=grad_magnitude,
                direction_change=direction_change,
            )
            self.gradient_stats[name] = stats
            
            # Compute stability score
            # s(p) = 1 / (1 + Δw × g_norm)
            stability = 1.0 / (1.0 + weight_change * grad_magnitude + 1e-8)
            
            # Smooth with previous score (EMA)
            alpha = 0.3
            if name in self.stability_scores:
                stability = alpha * stability + (1 - alpha) * self.stability_scores[name]
            
            self.stability_scores[name] = stability
            
            # Update previous state
            self.previous_weights[name] = current_weight.clone()
            self.previous_gradients[name] = current_grad.clone()
        
        # Adapt threshold based on gradient statistics
        self._adapt_threshold()
        
        return self.stability_scores.copy()
    
    def _adapt_threshold(self) -> None:
        """
        Adapt threshold based on training dynamics.
        
        τ = τ₀ + α × (σ_grad / μ_grad)
        
        When gradients are noisy (high variance), increase protection.
        When gradients are stable, allow more updates.
        """
        if not self.gradient_stats:
            return
        
        magnitudes = [s.magnitude for s in self.gradient_stats.values()]
        if not magnitudes:
            return
        
        mean_mag = sum(magnitudes) / len(magnitudes)
        if mean_mag < 1e-8:
            return
        
        variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
        std_mag = variance ** 0.5
        
        # Coefficient of variation
        cv = std_mag / (mean_mag + 1e-8)
        
        # Adapt threshold
        self.threshold = self.base_threshold + self.threshold_adaptation * cv
        self.threshold = min(max(self.threshold, 0.1), 0.9)  # Clamp
    
    def protect(self) -> Dict[str, float]:
        """
        Apply protection to gradients based on stability analysis.
        
        This modifies gradients in-place before optimizer.step().
        
        Returns:
            Dictionary of parameter names to protection rates applied
        """
        protection_rates = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            stability = self.stability_scores.get(name, 0.5)
            guard = self.guards.get(name)
            
            if guard is None:
                guard = LossGuard(self.threshold, self.soft_protection)
                self.guards[name] = guard
            
            # Update guard threshold
            guard.threshold = self.threshold
            
            # Apply protection
            original_grad = param.grad.data.clone()
            protected_grad = guard.evaluate(stability, param.grad.data)
            param.grad.data = protected_grad
            
            # Calculate protection rate
            original_norm = torch.norm(original_grad).item()
            protected_norm = torch.norm(protected_grad).item()
            
            if original_norm > 1e-8:
                protection_rate = 1.0 - (protected_norm / original_norm)
            else:
                protection_rate = 0.0
            
            protection_rates[name] = protection_rate
        
        # Update global statistics
        self.step_count += 1
        if protection_rates:
            avg_protection = sum(protection_rates.values()) / len(protection_rates)
            self.total_protection_rate = (
                (self.total_protection_rate * (self.step_count - 1) + avg_protection)
                / self.step_count
            )
        
        return protection_rates
    
    def get_stability_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of stability across all parameters.
        
        Returns:
            Dictionary with 'protected', 'neutral', 'volatile' percentages
        """
        if not self.stability_scores:
            return {'protected': 0.0, 'neutral': 0.0, 'volatile': 0.0}
        
        scores = list(self.stability_scores.values())
        total = len(scores)
        
        protected = sum(1 for s in scores if s > 0.7) / total
        volatile = sum(1 for s in scores if s < 0.3) / total
        neutral = 1.0 - protected - volatile
        
        return {
            'protected': protected * 100,
            'neutral': neutral * 100,
            'volatile': volatile * 100,
        }
    
    def get_state(self) -> Dict:
        """Get current state for logging or checkpointing."""
        return {
            'step_count': self.step_count,
            'threshold': self.threshold,
            'total_protection_rate': self.total_protection_rate,
            'stability_summary': self.get_stability_summary(),
        }
