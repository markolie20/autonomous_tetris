"""
Central place for all fixed hyper-parameters and magic numbers.
"""
# General
SEED = 0
BASELINE_EPISODES = 250
Q_LEARNING_EPISODES = 15_000
MAX_FRAMES = 10_000

# tetris_rl/config.py
VARIANTS = {
    # Drops ε from 1.0 → 0.90 each 10 000-frame episode
    # per-frame factor = 0.90 ** (1 / 10_000)  ≈ 0.99998946
    "q_fast_decay": dict(
        alpha       = 0.10,
        gamma       = 0.99,
        eps_start   = 1.0,
        eps_min     = 0.05,
        eps_decay   = 0.9999895,   # fast
        decay_after = 20_000       # ≈ 2 episodes
    ),

    # Drops ε from 1.0 → 0.995 each episode
    # per-frame factor = 0.995 ** (1 / 10_000) ≈ 0.9999995
    "q_slow_decay": dict(
        alpha       = 0.10,
        gamma       = 0.99,
        eps_start   = 1.0,
        eps_min     = 0.05,
        eps_decay   = 0.9999995,   # slow
        decay_after = 50_000       # ≈ 5 episodes
    ),

    # Same exploration schedule as slow-decay, but lower learning rate
    "q_low_lr": dict(
        alpha       = 0.05,        # smaller α
        gamma       = 0.99,
        eps_start   = 1.0,
        eps_min     = 0.05,
        eps_decay   = 0.9999995,
        decay_after = 50_000
    ),
}

PRINT_EVERY_TRAIN = 250
