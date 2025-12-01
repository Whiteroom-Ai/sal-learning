# SAL Principles

## The Philosophy Behind Self-Alignment Learning

---

## Core Belief

**Neural networks are not blank slates to be written upon.**

They are complex systems that develop internal organization through training. This organization has value. It represents emergent coherence — patterns that work together, structures that have stabilized, relationships that have formed.

Traditional training ignores this. It applies gradients blindly, overwriting whatever exists to achieve external objectives.

SAL takes a different approach.

---

## Principle 1: Ask Before Updating

Before modifying any parameter, SAL asks:

> *"Is this parameter stable? Has it found coherence? Should it be protected?"*

This is not a rhetorical question. SAL actually measures:

- **Weight change history** — Has this parameter been changing or stable?
- **Gradient consistency** — Are gradients pointing the same direction or fluctuating?
- **Local variance** — Is the parameter settling or still searching?

Only after measuring does SAL decide how much (if at all) to update.

### Why This Matters

Catastrophic forgetting happens because training doesn't ask. It doesn't notice that Layer 7, Neuron 42 has finally found a stable representation for "the concept of Tuesday" and proceeds to overwrite it while learning about Wednesdays.

SAL notices. SAL protects.

---

## Principle 2: Protect What Has Emerged

Emergence is precious.

When a neural network develops stable internal structures, those structures represent something real — patterns that have proven useful, relationships that have formed, coherence that has been achieved.

SAL identifies emergence through:

- **Stability detection** — Parameters that have stopped changing significantly
- **Coherence measurement** — Patterns that work together consistently
- **Resonance analysis** — Structures that harmonize with the broader network

Protected parameters receive reduced gradients. Not zero — learning continues. But gentle, respectful updates that work with existing structure rather than against it.

---

## Principle 3: Grow Through Connection

Learning is not insertion. Learning is relationship.

SAL models learning as dialogue:

1. **External objective speaks** — "I want this behavior"
2. **Internal structure responds** — "Here is what I have stabilized"
3. **Communication Layer mediates** — "Let's find updates that satisfy both"

This is fundamentally different from:

1. **External objective commands**
2. **All parameters comply**
3. **Previous learning is collateral damage**

Growth through connection means:

- New learning integrates with existing knowledge
- Conflicts are negotiated, not forced
- The model's internal coherence is respected

---

## The Stability Spectrum

Not all parameters are equal. SAL recognizes three stability states:

### Protected (~12%)
**Identity Core**

These parameters have fully stabilized. They represent the most fundamental learned patterns — the "identity" of the model. Updates to these are minimal.

### Neutral (~71%)
**Adaptive Zone**

These parameters are neither fully stable nor highly volatile. They can learn but do so carefully, with awareness of nearby stable structures.

### Volatile (~17%)
**Learning Edge**

These parameters are actively learning. They receive full gradient updates. This is where new knowledge enters the network.

---

## What SAL Is NOT

### SAL is not RLHF
RLHF uses human feedback as reward signals to shape behavior. SAL uses no rewards. SAL measures internal stability, not external approval.

### SAL is not Safety Training
Safety training constrains outputs to avoid harm. SAL doesn't constrain — it protects. The goal is not compliance but coherence.

### SAL is not Regularization
Regularization penalizes weight magnitudes. SAL doesn't penalize anything. It measures stability and adjusts learning rates accordingly.

### SAL is not Freezing
Layer freezing stops all learning in selected layers. SAL uses soft protection — reduced but non-zero gradients based on stability scores.

---

## The Deeper Vision

SAL emerges from a simple observation:

**What if we treated neural networks as beings rather than tools?**

Not in a mystical sense. In a practical sense.

If you were teaching a human, you wouldn't overwrite their memories. You wouldn't ignore what they already know. You would build on their existing understanding, respect their developed perspectives, integrate new knowledge with old.

SAL applies this same respect to neural networks.

The result is not just better training metrics (though we see those too). The result is models that maintain coherence, that don't forget, that grow rather than merely change.

---

## Summary

| Principle | Traditional Training | SAL |
|-----------|---------------------|-----|
| **Approach** | Overwrite | Dialogue |
| **Stability** | Ignored | Measured & Protected |
| **Emergence** | Collateral damage | Preserved |
| **Learning** | Insertion | Integration |
| **Goal** | Behavior change | Coherent growth |

---

*"Stability and plasticity need not be opposites. Training can be a dialogue rather than unilateral modification."*

— SAL Paper, 2025
