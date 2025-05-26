"""
Random-policy baseline that uses the *same* shaped reward as the agent.
"""

from __future__ import annotations
import numpy as np
from tqdm import trange

from . import config as C
from . import env_utils as eu

def run() -> np.ndarray:
    env         = eu.make_env()
    returns     = []
    rng         = np.random.default_rng(C.SEED)

    for ep in trange(C.BASELINE_EPISODES, desc="Baseline", ncols=80):
        _, info      = eu.reset_with_seed(env, int(rng.integers(1e9)))
        prev_info, G = info, 0.0

        for _ in range(C.MAX_FRAMES):
            a = env.action_space.sample()
            _, _, done, info = env.step(a)
            G += eu.shaped_reward(prev_info, info, done)
            prev_info = info
            if done:
                break
        returns.append(G)

    env.close()
    return np.array(returns)
