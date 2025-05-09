# Tetris Q-Learning Demo

---

## 1 · Project Overview
This repository shows a **minimal, reproducible Q-learning agent** for the classic NES-Tetris environment from `gym-tetris`.

| Agent | Purpose |
|-------|---------|
| **Random policy** | Baseline that plays by sampling random actions (with hard-drop every other frame to finish episodes quickly). |
| **Q-learning agent** | Learns a tabular Q-table on a compact state `(current piece, board height, well-depth bucket)` and receives a shaped dense reward at every piece-lock or line-clear. |

The notebook (or script) prints episodic returns, decays ε, and plots:

1. **Advantage** (Q-learning return − random-baseline mean).  
2. **Dense episode returns** for both agents.

---

## 2 · Installation
> Tested on **Python ≥ 3.9** with **gym-tetris ≈ 5.0** and **nes-py ≈ 9.1**.

```bash
# (Optional) create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# Install core requirements
pip install --upgrade pip
pip install gym-tetris matplotlib numpy

## 4 · Running the Demo
```bash
jupyter notebook tetris_qlearning.ipynb
```
