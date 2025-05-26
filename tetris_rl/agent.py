"""
Tabular ε-greedy Q-learning agent stored in a defaultdict.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
import os, pickle

from . import config as C
from . import env_utils as eu

class QLearningAgent:
    def __init__(self, rng: np.random.Generator, **hp):
        # pull hyper-parameters from dict, fall back to sensible defaults
        self.alpha        = hp.get("alpha", 0.10)
        self.gamma        = hp.get("gamma", 0.99)
        self.eps          = hp.get("eps_start", 1.0)
        self.eps_min      = hp.get("eps_min", 0.05)
        self.eps_decay    = hp.get("eps_decay", 0.995)
        self.decay_after  = hp.get("decay_after", 10_000)

        self.Q   = defaultdict(lambda: np.zeros(eu.N_ACTIONS, dtype=np.float32))
        self.rng = rng
        self.frames_seen = 0

    # ─────────────────────────── public helpers ───────────────────────────────
    def select_action(self, state):
        if self.rng.random() < self.eps:
            return self.rng.integers(eu.N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, terminated):
        best_next = 0.0 if terminated else np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    # ─────────────────────────── training loop per episode ────────────────────
    def play_episode(self, env):
        """
        Run one episode and return its cumulative shaped reward G.
        ε is updated ONCE per episode, after the final frame.
        """
        # ─────────────────── episode init ───────────────────
        _, info  = eu.reset_with_seed(env, int(self.rng.integers(1e9)))
        state    = eu.state_from_info(env, info)
        prev_info, G = info, 0.0

        # ─────────────────── main frame loop ────────────────
        for _ in range(C.MAX_FRAMES):
            a = self.select_action(state)

            _, _, done, info = env.step(int(a))
            self.frames_seen += 1                      # global frame counter

            r = eu.shaped_reward(prev_info, info, done)
            G += r

            # Q-update
            if done:
                self.update(state, a, r, None, True)
                break
            else:
                state_next = eu.state_from_info(env, info)
                self.update(state, a, r, state_next, False)
                state, prev_info = state_next, info

        # ─────────────────── ε schedule (once per episode) ───────────────
        if self.frames_seen > self.decay_after:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        return G

    
    def save(self, path:str):
        """Pickle the learned Q-table + hyper-params for later reload."""
        payload = {
            "Q"   : {k: v.copy() for k, v in self.Q.items()},   # dict → np arrays
            "hp"  : dict(alpha=self.alpha, gamma=self.gamma,
                         eps_start=self.eps, eps_min=self.eps_min,
                         eps_decay=self.eps_decay, decay_after=self.decay_after),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
