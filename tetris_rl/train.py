from __future__ import annotations
import os, numpy as np
from tqdm import trange, tqdm
from . import env_utils as eu, config as C, agent as ag

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# ────────────────────────────────────────────────────────────────────────
def train_variant(
    name: str,
    hp  : dict,
    rng,
    env = None,                 # if provided by caller, re-use; else create
) -> np.ndarray:
    """
    Train one Q-learning agent and return its per-episode returns.

    Parameters
    ----------
    name : str       – variant label
    hp   : dict      – hyper-parameters passed to QLearningAgent
    rng             – numpy.random.Generator
    env  : gym.Env | None
        If None, a fresh env is created and closed internally.

    Side-effects
    ------------
    • Saves pickle  results/<variant>_model.pkl
    • Returns np.ndarray of length C.Q_LEARNING_EPISODES
    """
    own_env = env is None
    if own_env:
        env = eu.make_env(skip=8)         # default frame-skip

    learner = ag.QLearningAgent(rng, **hp)
    returns = []

    for ep in trange(
        C.Q_LEARNING_EPISODES,
        desc=f"Training ({name})",
        ncols=80,
        leave=False,
    ):
        G = learner.play_episode(env)
        returns.append(G)

        if (ep + 1) % C.PRINT_EVERY_TRAIN == 0:
            mean_k = np.mean(returns[-C.PRINT_EVERY_TRAIN:])
            tqdm.write(
                f"{name:<12} | "
                f"ep {ep+1:4d}/{C.Q_LEARNING_EPISODES} | "
                f"ε={learner.eps:5.3f} | "
                f"µ{C.PRINT_EVERY_TRAIN:02d}={mean_k:8.2f}"
            )

    if own_env:
        env.close()

    # save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    learner.save(os.path.join(MODELS_DIR, f"{name}_model.pkl"))

    return np.asarray(returns, dtype=np.float32)

