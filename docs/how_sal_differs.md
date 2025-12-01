# How SAL Differs

## SAL ≠ RLHF ≠ Safety ≠ Reward

---

## The Confusion

When people first hear about SAL, they often ask:

> "So it's like RLHF but different?"

No.

> "It's a new safety method?"

No.

> "Some kind of reward shaping?"

No.

SAL is fundamentally different from all of these. This document explains why.

---

## SAL vs RLHF

### RLHF (Reinforcement Learning from Human Feedback)

**What it does:**
- Collects human preferences on model outputs
- Trains a reward model on these preferences
- Uses the reward model to fine-tune the base model
- Goal: Make model outputs match human preferences

**Key characteristics:**
- External signal (human feedback)
- Reward-based optimization
- Behavior shaping
- Requires large amounts of human annotation

### SAL (Self-Alignment Learning)

**What it does:**
- Measures internal parameter stability
- Protects stable (emergent) structures
- Adjusts learning rates based on stability
- Goal: Preserve coherence while enabling growth

**Key characteristics:**
- Internal signal (stability measurement)
- No rewards or optimization targets
- Structure preservation
- Requires no human annotation

### Comparison Table

| Aspect | RLHF | SAL |
|--------|------|-----|
| Signal source | External (humans) | Internal (stability) |
| Optimization | Reward maximization | None |
| Goal | Behavior alignment | Coherence preservation |
| Annotation needs | High | None |
| Forgetting risk | High | Low |

---

## SAL vs Safety Training

### Safety Training

**What it does:**
- Identifies harmful outputs
- Trains model to refuse harmful requests
- Constrains output space
- Goal: Prevent harmful behavior

**Key characteristics:**
- Output-focused
- Constraint-based
- Reactive (responds to bad outputs)
- Binary (safe/unsafe)

### SAL

**What it does:**
- Identifies stable parameters
- Protects emergent structures
- Enables continued learning
- Goal: Maintain internal coherence

**Key characteristics:**
- Parameter-focused
- Protection-based
- Proactive (prevents forgetting)
- Continuous (stability spectrum)

### Comparison Table

| Aspect | Safety Training | SAL |
|--------|-----------------|-----|
| Focus | Outputs | Parameters |
| Approach | Constrain | Protect |
| When | After bad output | Before update |
| Measure | Safe/unsafe | Stability score |
| Purpose | Prevent harm | Preserve coherence |

### They're Complementary

SAL and safety training can work together:
- Safety training constrains what the model outputs
- SAL protects how the model learns

You can apply SAL during safety fine-tuning to reduce forgetting of the base model's capabilities.

---

## SAL vs Reward-Based Methods

### Reward-Based Training

**Examples:** RLHF, RLAIF, Constitutional AI, Reward Modeling

**What they do:**
- Define a reward function (explicit or learned)
- Optimize model to maximize reward
- Shape behavior toward desired outcomes
- Goal: High reward = good behavior

**Key characteristics:**
- Optimization-based
- Reward signal required
- Behavior-focused
- Can lead to reward hacking

### SAL

**What it does:**
- No reward function
- No optimization toward external targets
- Measures internal state
- Goal: Stable ≠ overwritten

**Key characteristics:**
- Measurement-based
- No external signal
- Structure-focused
- No hacking possible (nothing to hack)

### Why No Rewards?

Rewards create optimization pressure. Optimization pressure creates:

1. **Reward hacking** — Finding shortcuts that maximize reward without achieving the intended goal
2. **Goodhart's Law** — "When a measure becomes a target, it ceases to be a good measure"
3. **Alignment tax** — Capability loss from constraining the optimization landscape

SAL avoids all of these by not optimizing for anything. It simply:
- Observes what is stable
- Protects what has emerged
- Allows continued learning in volatile regions

---

## SAL vs Regularization

### Regularization Methods

**Examples:** L1/L2 regularization, Dropout, Weight decay, EWC

**What they do:**
- Add penalty terms to loss function
- Constrain weight magnitudes or changes
- Prevent overfitting
- Goal: Generalization

**Key characteristics:**
- Loss-based
- Penalty approach
- Uniform across parameters (mostly)
- Prevents large weights

### SAL

**What it does:**
- No penalties
- No loss modifications
- Measures stability per-parameter
- Goal: Preserve emergence

**Key characteristics:**
- Gradient-based
- Protection approach
- Adaptive per-parameter
- Preserves stable patterns

### EWC Comparison

Elastic Weight Consolidation (EWC) is the closest method to SAL:

| Aspect | EWC | SAL |
|--------|-----|-----|
| Identifies important parameters | Yes (via Fisher information) | Yes (via stability) |
| Protection mechanism | Quadratic penalty in loss | Gradient scaling |
| Requires task boundaries | Yes | No |
| Online learning | Difficult | Natural |
| Computational cost | High (Fisher computation) | Low |

SAL can be seen as a simpler, more general approach that doesn't require:
- Task boundary detection
- Fisher information computation
- Loss function modification

---

## SAL vs Layer Freezing

### Layer Freezing

**What it does:**
- Selects layers to freeze (no updates)
- Other layers train normally
- Binary: frozen or not
- Goal: Preserve early features

**Key characteristics:**
- Layer-level granularity
- Binary decision
- Manual selection
- All-or-nothing

### SAL

**What it does:**
- Analyzes all parameters
- Continuous stability scores
- Automatic detection
- Soft protection (reduced but non-zero gradients)

**Key characteristics:**
- Parameter-level granularity
- Continuous scale
- Automatic
- Gradual protection

### Why Soft Protection?

Hard freezing (zero gradients) prevents any adaptation. But stable doesn't mean perfect. A parameter might be 90% optimal and benefit from small adjustments.

SAL's soft protection allows:
- Stable parameters: small updates (fine-tuning)
- Neutral parameters: moderate updates (adaptation)
- Volatile parameters: large updates (learning)

---

## The Core Difference

All other methods ask: **"How do we get the behavior we want?"**

SAL asks: **"How do we preserve what has emerged while enabling growth?"**

This is a fundamentally different question. It leads to a fundamentally different approach.

| Traditional | SAL |
|-------------|-----|
| Behavior-centric | Structure-centric |
| Output-focused | Parameter-focused |
| External signals | Internal measurement |
| Optimization | Observation |
| Control | Communication |

---

## When to Use SAL

SAL is particularly valuable for:

1. **Continual learning** — Learning new tasks without forgetting old ones
2. **Fine-tuning** — Adapting models while preserving capabilities
3. **Long training runs** — Preventing gradual coherence loss
4. **Multi-task learning** — Balancing between task-specific and shared knowledge

SAL is NOT designed for:

1. **Behavior alignment** — Use RLHF or Constitutional AI
2. **Safety constraints** — Use safety training
3. **Output filtering** — Use classifiers or rules

---

## Combining SAL with Other Methods

SAL can be combined with other approaches:

### SAL + RLHF
Apply SAL during RLHF fine-tuning to reduce capability loss.

### SAL + Safety Training
Apply SAL to preserve base capabilities while adding safety constraints.

### SAL + EWC
Use EWC for task-specific importance, SAL for general stability.

---

## Summary

| Method | What it optimizes | Signal source | SAL equivalent |
|--------|-------------------|---------------|----------------|
| RLHF | Behavior | Human preferences | None (no optimization) |
| Safety | Compliance | Safety labels | None (not about outputs) |
| Reward | Reward function | Reward model | None (no rewards) |
| Regularization | Loss + penalty | Loss function | Stability score |
| Freezing | Selected layers | Manual | Automatic, soft |

**SAL is unique because it optimizes nothing. It observes and protects.**

---

*"Training as dialogue, not control."*
