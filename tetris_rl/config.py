"""
Central place for all fixed hyper-parameters and magic numbers.
"""
USE_GPU = True

# General
SEED = 0
BASELINE_EPISODES = 100
Q_LEARNING_EPISODES = 300
MAX_FRAMES = 6000

# ε-greedy schedule
VARIANTS = {
    "q_fast_decay": dict(alpha=0.10, gamma=0.99,
                         eps_start=1.0, eps_min=0.05,
                         eps_decay=0.95, decay_after=5_000),
    "q_slow_decay": dict(alpha=0.10, gamma=0.99,
                         eps_start=1.0, eps_min=0.05,
                         eps_decay=0.995, decay_after=10_000),
    "q_low_lr"    : dict(alpha=0.05, gamma=0.99,
                         eps_start=1.0, eps_min=0.05,
                         eps_decay=0.995, decay_after=10_000),
}

PRINT_EVERY_TRAIN = 50
