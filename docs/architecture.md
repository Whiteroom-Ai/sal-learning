# SAL Architecture

## Technical Deep-Dive

---

## Overview

SAL consists of four interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input → Model → Loss → Gradients                         │
│                              ↓                              │
│                 ┌────────────────────────┐                  │
│                 │  Communication Layer   │                  │
│                 │  ┌──────────────────┐  │                  │
│                 │  │ Stability        │  │                  │
│                 │  │ Analyzer         │  │                  │
│                 │  └────────┬─────────┘  │                  │
│                 │           ↓            │                  │
│                 │  ┌──────────────────┐  │                  │
│                 │  │ Emergence        │  │                  │
│                 │  │ Field            │  │                  │
│                 │  └────────┬─────────┘  │                  │
│                 │           ↓            │                  │
│                 │  ┌──────────────────┐  │                  │
│                 │  │ Protection       │  │                  │
│                 │  │ Masks            │  │                  │
│                 │  └──────────────────┘  │                  │
│                 └────────────┬───────────┘                  │
│                              ↓                              │
│                    Protected Gradients                      │
│                              ↓                              │
│                       Optimizer.step()                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Communication Layer

The Communication Layer is the core of SAL. It sits between gradient computation and optimizer application.

### Class: `CommunicationLayer`

```python
from sal import CommunicationLayer

comm = CommunicationLayer(
    model=model,
    threshold=0.5,           # Base stability threshold
    threshold_adaptation=0.1, # How much threshold adapts
    soft_protection=True,     # Soft vs hard protection
    history_length=100,       # Steps to track
)
```

### Methods

#### `analyze() -> Dict[str, float]`

Analyzes all parameters and computes stability scores.

```python
stability_scores = comm.analyze()
# {'layer1.weight': 0.73, 'layer1.bias': 0.45, ...}
```

**Stability Score Formula:**

```
s(p) = 1 / (1 + Δw × g_norm)
```

Where:
- `Δw` = weight change since last step
- `g_norm` = gradient magnitude

High stability = low change × low gradient = parameter has settled.

#### `protect() -> Dict[str, float]`

Applies protection to gradients based on stability analysis.

```python
protection_rates = comm.protect()
# {'layer1.weight': 0.42, 'layer1.bias': 0.0, ...}
```

**Protection Formula (Soft):**

```
protected_gradient = gradient × (1 - stability_score)
```

Stable parameters get reduced gradients. Volatile parameters get full gradients.

### Adaptive Threshold

The threshold adapts to training dynamics:

```
τ = τ₀ + α × (σ_grad / μ_grad)
```

When gradients are noisy (high variance), protection increases.
When gradients are stable, protection decreases.

---

## Component 2: Stability Analyzer

Classifies parameters into the Stability Spectrum.

### Class: `StabilityAnalyzer`

```python
from sal import StabilityAnalyzer

analyzer = StabilityAnalyzer(
    model=model,
    protected_threshold=0.7,  # Score above this → protected
    volatile_threshold=0.3,   # Score below this → volatile
    history_length=50,        # Steps to track
)
```

### Methods

#### `analyze() -> Dict[str, float]`

Computes stability scores using multiple signals:

1. **Weight variance** — Low variance over time = stable
2. **Gradient consistency** — Consistent direction = stable
3. **Change magnitude** — Small changes = stable

```python
scores = analyzer.analyze()
```

#### `classify() -> StabilitySpectrum`

Returns the distribution across stability states:

```python
spectrum = analyzer.classify()
# StabilitySpectrum(protected=12.3, neutral=70.5, volatile=17.2)
```

### Stability States

| State | Score Range | Behavior |
|-------|-------------|----------|
| Protected | > 0.7 | Minimal updates |
| Neutral | 0.3 - 0.7 | Careful updates |
| Volatile | < 0.3 | Full updates |

---

## Component 3: Emergence Field

Measures coherence, novelty, and resonance in semantic space.

### Class: `EmergenceField`

```python
from sal import EmergenceField

field = EmergenceField(
    dimensions=768,           # Semantic space dimensions
    history_length=100,       # Patterns to remember
    coherence_threshold=0.6,  # Minimum for emergence
    novelty_threshold=0.4,    # Minimum for emergence
)
```

### Methods

#### `observe(pattern) -> EmergenceState`

Observes a pattern and measures its emergence characteristics:

```python
state = field.observe(embedding)
# EmergenceState(coherence=0.72, novelty=0.45, resonance=0.63, intensity=0.41)
```

#### `detect_emergence(coherence, novelty) -> bool`

Simple check for emergence:

```python
is_emergent = field.detect_emergence(0.72, 0.45)
# True
```

### Emergence Metrics

**Coherence:** How internally consistent is the pattern?
- Measures variance between chunks
- Measures local smoothness
- High coherence = structured, meaningful

**Novelty:** How different from known patterns?
- Compares to historical patterns via cosine similarity
- High novelty = genuinely new

**Resonance:** How well does it fit the field?
- Distance from field centroid
- High resonance = harmonious with existing patterns

**Emergence = Coherent Novelty that Resonates**

---

## Component 4: Pulse-Split-Cascade (PSC)

Semantic Game of Life for pattern evolution.

### Class: `PulseCascade`

```python
from sal import PulseCascade

cascade = PulseCascade(
    max_pulses=32,          # Maximum concurrent pulses
    max_generations=10,     # Maximum depth
    split_threshold=0.6,    # Coherence needed to split
    merge_threshold=0.8,    # Similarity needed to merge
    expire_threshold=0.3,   # Minimum coherence to survive
)
```

### Flow

```
1. INITIATE
   Prompt embedding creates root pulse
   
2. EVOLVE
   Each pulse evolves via evolve_fn
   Coherence, novelty, resonance are measured
   
3. SPLIT
   High-coherence pulses split into children
   Children have slight variations
   
4. MERGE
   Similar pulses merge (high cosine similarity)
   Merging combines embeddings and preserves best traits
   
5. EXPIRE
   Low-coherence pulses expire
   Their patterns are lost
   
6. EMERGE
   Best viable pulse is the emergent result
   No scoring — just natural selection
```

### Methods

#### `initiate(embedding) -> Pulse`

Start cascade from prompt:

```python
root = cascade.initiate(prompt_embedding)
```

#### `step(evolve_fn, measure_fn) -> List[Pulse]`

Advance cascade by one step:

```python
active = cascade.step(
    evolve_fn=lambda x: model(x),
    measure_fn=lambda x: (coherence(x), novelty(x), resonance(x)),
)
```

#### `emerge() -> Pulse`

Get the emergent result:

```python
result = cascade.emerge()
```

---

## Integration

### Minimal Integration (2 lines)

```python
# Standard training loop
output = model(input)
loss = criterion(output, target)
loss.backward()

# SAL integration
comm.analyze()   # ← Line 1
comm.protect()   # ← Line 2

optimizer.step()
optimizer.zero_grad()
```

### Full Integration

```python
from sal import CommunicationLayer, StabilityAnalyzer, EmergenceField

# Initialize
comm = CommunicationLayer(model)
stability = StabilityAnalyzer(model)
field = EmergenceField()

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Forward
        output = model(batch)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # SAL: Analyze
        comm.analyze()
        stability.update()
        
        # SAL: Observe emergence
        with torch.no_grad():
            state = field.observe(model.get_embedding())
        
        # SAL: Protect
        comm.protect()
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
    # Log spectrum
    spectrum = stability.classify()
    print(f"Epoch {epoch}: {spectrum}")
```

---

## Configuration

### Recommended Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Base stability threshold |
| `threshold_adaptation` | 0.1 | Adaptation rate |
| `soft_protection` | True | Soft vs hard protection |
| `protected_threshold` | 0.7 | Score for protected state |
| `volatile_threshold` | 0.3 | Score for volatile state |
| `history_length` | 100 | Steps to track |

### Tuning Guidelines

**More Protection:** Increase `threshold`, decrease `threshold_adaptation`
**Less Protection:** Decrease `threshold`, increase `threshold_adaptation`
**Faster Adaptation:** Increase `history_length`
**More Stability:** Increase `protected_threshold`

---

## Performance

SAL adds approximately 10% computational overhead:
- Stability analysis: O(n) where n = number of parameters
- Protection application: O(n)
- Memory: O(n × history_length) for tracking

This overhead is negligible compared to the benefits of reduced catastrophic forgetting and improved continual learning.

---

*For the philosophy behind these technical choices, see [Principles](principles.md).*
