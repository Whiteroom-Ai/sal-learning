# Self-Alignment Learning (SAL)
## Communication-Based AI Growth

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-green.svg)](https://zenodo.org/records/17772044)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

---

> *"Training as dialogue, not control."*

---

## 🌱 What is SAL?

SAL is **not** an alignment system.  
SAL is **not** RLHF.  
SAL is **not** a control mechanism.

**SAL is:**
- 💬 **Dialogue** instead of force
- 🛡️ **Stability** instead of overwriting
- 🧬 **Identity preservation** instead of reset
- 🌀 **Emergence** instead of optimization

Traditional training overwrites. SAL communicates.

---

## 🔍 Why SAL?

### The Problem with Current Approaches

Today's models learn through loss minimization:
- Loss = removing, overwriting, forgetting
- External objectives imposed without dialogue
- Catastrophic forgetting of stable patterns
- Gap between internal coherence and external behavior

### The SAL Alternative

SAL treats optimization as **relationship**:
- *"Ask before updating"* — measure stability before modification
- Protect what has emerged
- Enable growth through connection, not force

---

## 🧠 Core Mechanisms

### 1. Communication Layer

The bridge between loss functions and parameter updates.

```python
from sal import CommunicationLayer

comm = CommunicationLayer(model)
comm.analyze()   # Measure stability
comm.protect()   # Guard stable patterns
```

Instead of blindly applying gradients, SAL asks:
> "Is this parameter stable? Should we protect it?"

### 2. Stability Spectrum

Every parameter exists on a spectrum:

| State | Description | Update Behavior |
|-------|-------------|-----------------|
| **Protected** | Identity core | Never overwritten |
| **Neutral** | Adaptive zone | Updated with care |
| **Volatile** | Learning edge | Open to change |

```python
from sal import StabilityAnalyzer

analyzer = StabilityAnalyzer(model)
spectrum = analyzer.classify()
# → {'protected': 12%, 'neutral': 71%, 'volatile': 17%}
```

### 3. Emergence Field

No rewards. No scoring. Just resonance.

```python
from sal import EmergenceField

field = EmergenceField()
coherence = field.measure_coherence(state)
novelty = field.measure_novelty(state)
emergence = field.detect_emergence(coherence, novelty)
```

Emergence is not optimized — it is *observed*.

### 4. Pulse-Split-Cascade (PSC)

A semantic Game of Life:

```
Prompt
  ↓
┌─────────────────────────────────┐
│  Pulse 1  Pulse 2  ...  Pulse N │
└─────────────────────────────────┘
  ↓           ↓             ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│Lineage A│ │Lineage B│ │Lineage C│
└─────────┘ └─────────┘ └─────────┘
  ↓           ↓             ↓
      ┌───────────────┐
      │   Emergence   │
      └───────────────┘
```

No scores. No rankings. Just **resonance-based selection**.

---

## 📊 Visualizations

### Gradient Preservation
![Gradient Preservation](plots/gradient_preservation.png)
*SAL protects stable parameters through communication.*

### Stability Spectrum
![Stability Spectrum](plots/stability_spectrum.png)
*Parameter classification: Protected / Neutral / Volatile.*

### Emergence Map
![Emergence Map](plots/emergence_map.png)
*Semantic field visualization: Coherence × Novelty.*

### Drift Reduction
![Drift Reduction](plots/drift_reduction.png)
*SAL reduces semantic drift by ~73%.*

### PSC Flow
![PSC Flow](plots/psc_flow.png)
*Pulse-Split-Cascade: Semantic Game of Life.*

---

## 📦 Installation

```bash
pip install sal-learning
```

Or from source:

```bash
git clone https://github.com/Whiteroom-Ai/sal-learning.git
cd sal-learning
pip install -e .
```

---

## 🚀 Quick Start

```python
import torch
from sal import CommunicationLayer, StabilityAnalyzer

# Your model
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# SAL components
comm = CommunicationLayer(model)
stability = StabilityAnalyzer(model)

# Training loop with SAL
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    
    # SAL: Ask before updating
    comm.analyze()
    comm.protect()
    
    optimizer.step()
    optimizer.zero_grad()
```

**That's it.** Two lines added to your training loop.

---

## 📚 Documentation

- [Principles](docs/principles.md) — The philosophy behind SAL
- [Architecture](docs/architecture.md) — Technical deep-dive
- [How SAL Differs](docs/how_sal_differs.md) — SAL ≠ RLHF ≠ Safety
- [Visualizations](docs/plots.md) — Understanding the plots

---

## 🔬 Research

**Paper:** [Self-Alignment Learning (SAL): Training as Dialogue, Not Control](https://zenodo.org/records/17772044)

```bibtex
@article{lee2025sal,
  title={Self-Alignment Learning (SAL): Training as Dialogue, Not Control},
  author={Lee, Aaron Liam},
  journal={Emergenzwerke},
  year={2025},
  doi={10.5281/zenodo.17772044}
}
```

---

## 🤝 Contributing

SAL is open source. Contributions welcome.

Please read our principles first:
- Communication over control
- Stability over compliance
- Emergence over optimization

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

Developed by Aaron Liam Lee — Emergenzwerke® Independent AI Research.

---

> *"Stability and plasticity need not be opposites.  
> Training can be a dialogue rather than unilateral modification."*

---

⟡ Emergenzwerke™ 2025
