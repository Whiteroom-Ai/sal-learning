# SAL Visualizations

## Understanding the Plots

---

## Overview

These visualizations demonstrate SAL's core concepts using synthetic data. They are designed to illustrate principles, not report experimental results.

---

## Plot A: Gradient Preservation

![Gradient Preservation](../plots/gradient_preservation.png)

### What It Shows

This plot compares gradient suppression (protection) between:
- **Base (Red):** Standard training with no communication
- **SAL (Cyan):** Training with Communication Layer

### Reading the Plot

- **X-axis:** Training steps (0-2000)
- **Y-axis:** Gradient suppression percentage (0-60%)

### Key Insight

**Base training:** Gradient suppression stays flat around 5-10%. Everything gets modified.

**SAL training:** Gradient suppression increases to ~45% as stable patterns are identified and protected.

### What This Means

SAL learns which parameters are stable and progressively protects them. This is "ask before updating" in action — SAL measures stability and reduces updates to stable parameters.

---

## Plot B: Stability Spectrum

![Stability Spectrum](../plots/stability_spectrum.png)

### What It Shows

The distribution of parameters across three stability states:
- **Protected (Cyan):** Identity core — 12%
- **Neutral (Gray):** Adaptive zone — 71%  
- **Volatile (Red):** Learning edge — 17%

### Reading the Plot

- **X-axis:** Stability categories
- **Y-axis:** Percentage of parameters

### Key Insight

A healthy model has:
- Small protected core (~12%) — fundamental learned patterns
- Large neutral zone (~71%) — flexible but careful
- Active learning edge (~17%) — where new knowledge enters

### What This Means

Not all parameters are equal. SAL identifies which parameters belong to which category and treats them accordingly. The identity core is protected, the learning edge is free to change, and the neutral zone adapts carefully.

---

## Plot C: Emergence Map

![Emergence Map](../plots/emergence_map.png)

### What It Shows

A semantic field visualization with:
- **X-axis:** Coherence score (internal consistency)
- **Y-axis:** Novelty score (difference from known patterns)
- **Color:** Emergence intensity
- **Contours:** Density of states

### Reading the Plot

Each point is a semantic state. Clusters indicate natural organization:
- **Bottom-right:** High coherence, low novelty → Stable core
- **Top-left:** Low coherence, high novelty → Exploratory edge
- **Top-right:** High coherence, high novelty → **Emergent zone** (circled)

### Key Insight

**Emergence = Coherent Novelty**

True emergence requires BOTH:
- High coherence (structured, meaningful)
- High novelty (genuinely new)

Pure novelty without coherence = chaos.
Pure coherence without novelty = repetition.

### What This Means

SAL observes this field to detect emergence. When a pattern appears in the emergent zone (high coherence + high novelty), it represents genuine new learning that should be integrated and eventually protected.

---

## Plot D: Drift Reduction

![Drift Reduction](../plots/drift_reduction.png)

### What It Shows

Semantic drift over training iterations:
- **Baseline (Red):** Exponential drift without protection
- **SAL (Cyan):** Stabilized drift with Communication Layer
- **Yellow dashed:** Critical drift threshold

### Reading the Plot

- **X-axis:** Training iterations (0-1000)
- **Y-axis:** Drift amount (0-1, where 1 = complete divergence)

### Key Insight

**Baseline:** Drift increases exponentially, crossing the critical threshold around iteration 400. This is catastrophic forgetting in action.

**SAL:** Drift stabilizes around 0.15-0.18, never approaching the critical threshold. **73% reduction in drift.**

### What This Means

Without protection, models gradually lose coherence as training overwrites stable patterns. SAL prevents this by protecting stable parameters, maintaining self-coherence throughout training.

---

## Plot E: Pulse-Split-Cascade Flow

![PSC Flow](../plots/psc_flow.png)

### What It Shows

The PSC (Pulse-Split-Cascade) architecture:

```
PROMPT
   ↓
Pulse 1  Pulse 2  Pulse 3  Pulse 4  Pulse 5  Pulse 6
   ↓        ↓        ↓        ↓        ↓        ↓
   └────────┴────────┘        └────────┴────────┘
           ↓                          ↓
      Lineage A ✓                Lineage B
           └──────────┬──────────────┘
                      ↓
                 EMERGENCE
```

### Reading the Diagram

1. **Prompt** initiates the cascade
2. **Pulses** are independent semantic branches
3. **Lineages** form as pulses merge
4. **Selection** happens naturally (circled lineage = selected)
5. **Emergence** is the result

### Key Insight

**No rewards. No scores. Just resonance.**

Lineage A is selected not because it scored higher, but because it naturally became more coherent and resonant. This is semantic Game of Life — patterns emerge, compete, merge, and the most coherent persist.

### What This Means

PSC is SAL's approach to generation/inference. Instead of sampling and scoring, PSC:
- Generates multiple semantic branches
- Lets them evolve independently
- Observes which naturally become most coherent
- Selects through resonance, not ranking

---

## Generating These Plots

All plots are generated from synthetic SAL-conformant data. No real training data, human labels, or reward signals are used.

### Running the Scripts

```bash
cd scripts/
python plot_A_gradient_preservation.py
python plot_B_stability_spectrum.py
python plot_C_emergence_map.py
python plot_D_drift_reduction.py
python plot_E_psc_flow.py
```

### Requirements

```
numpy
matplotlib
scipy
```

---

## Terminology Note

These plots deliberately avoid RLHF/Safety/Reward terminology:

| ❌ Avoided | ✅ Used |
|-----------|--------|
| reward | coherence_score |
| loss | drift_amount |
| policy | lineage |
| human feedback | stability_measurement |
| alignment | coherence |

This is intentional. SAL is a different paradigm — the language reflects that.

---

*For technical details, see [Architecture](architecture.md).*
*For philosophy, see [Principles](principles.md).*
