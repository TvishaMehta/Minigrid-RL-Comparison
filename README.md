# Comparative Analysis of Reinforcement Learning Algorithms on MiniGrid

This repository presents an implementation and comparison of classical
tabular reinforcement learning algorithms on the MiniGrid environment as
part of the IvLabs Summer Internship selection task.

The goal is to study and compare the learning behavior of different RL
algorithms in a discrete state–action space.

---

## Algorithms Implemented

- Monte Carlo Control (Every-Visit, ε-greedy)
- Q-learning (off-policy temporal difference control)
- SARSA(λ) (on-policy temporal difference control with eligibility traces)

Each algorithm learns an optimal policy for navigating MiniGrid environments
using a Q-table representation.

---

## Environment

- MiniGrid (Gymnasium)
- Discrete observation and action space
- Sparse reward setting

---

## Results

Training performance was evaluated using episode reward curves.

### Monte Carlo Control

![Monte Carlo](monte%20carlo.png)

### Q-learning

![Q-learning](q%20learning.png)

### SARSA(λ)

![SARSA Lambda](sarsa%20lambda.png)

These plots illustrate differences in convergence behavior, stability, and
sample efficiency across algorithms.

---

## SARSA(λ) Decay Strategy Comparison

To understand how different exploration decay strategies influence
SARSA(λ), we compare reward curves under multiple epsilon decay schedules.

| Decay Strategy | Curve Variant 1 | Curve Variant 2 | Interpretation |
|----------------|----------------|----------------|----------------|
| **Linear decay** | ![Linear Decay 1](sarsa_lambda_linear_decay_1.png) | ![Linear Decay 2](sarsa_lambda_linear_decay_2.png) | Gradual reduction in exploration creates a smooth learning progression with moderate stability and balanced exploration/exploitation. |
| **Exponential decay** | ![Exp Decay 1](sarsa_lambda_exp_decay_1.png) | ![Exp Decay 2](sarsa_lambda_exp_decay_2.png) | Rapid decay accelerates exploitation early, potentially improving initial convergence but risking suboptimal policy locking. |
| **Inverse time decay** | ![Inverse Decay 1](sarsa_lambda_inverse_decay_1.png) | ![Inverse Decay 2](sarsa_lambda_inverse_decay_2.png) | Slow reduction in exploration supports continued learning and adaptation, often producing stable long-term performance. |

### Interpretation Highlights

- **Linear decay** leads to balanced exploration and exploitation, smoothing the reward curve.
- **Exponential decay** encourages faster exploitation, which can help early performance but may miss optimal policy exploration.
- **Inverse time decay** keeps exploration higher longer, often yielding a more stable long-term increase in rewards.

---

## Tech Stack

- Python
- Gymnasium / MiniGrid
- NumPy
- Matplotlib

---

## How to Run

Install dependencies:
```bash
pip install gymnasium minigrid numpy matplotlib
