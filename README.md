# Comparative Analysis of Reinforcement Learning Algorithms on MiniGrid

This repository presents an implementation and comparison of classical **tabular reinforcement learning algorithms**
on the **MiniGrid** environment as part of the IvLabs Summer Internship selection task.

The goal is to study and compare the learning behavior of different RL algorithms in a discrete state–action space.

---

## Algorithms Implemented

- **Monte Carlo Control (Every-Visit, ε-greedy)**
- **Q-learning** (off-policy temporal difference control)
- **SARSA(λ)** (on-policy temporal difference control with eligibility traces)

Each algorithm learns an optimal policy for navigating MiniGrid environments using a Q-table representation.

---

## Environment

- **MiniGrid** (Gymnasium)
- Discrete observation and action space
- Sparse reward setting

---

## Results

Training performance was evaluated using **episode reward curves**.

### Monte Carlo Control
![Monte Carlo](monte%20carlo.png)

### Q-learning
![Q-learning](q%20learning.png)

### SARSA(λ)
![SARSA Lambda](sarsa%20lambda.png)

These plots illustrate differences in convergence behavior, stability, and sample efficiency across algorithms.

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
