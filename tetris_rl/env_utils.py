"""
Environment creation + all helper functions that depend on the Tetris env.
Compatible with gym-tetris 3.0.4 (old Gym API).
"""

from __future__ import annotations
import random, numpy as np, gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from typing import Tuple, Dict
import time

# tetris_rl/env_utils.py
from .frame_skip import FrameSkip          # add this import

def make_env(skip: int = 8, delay_ms: int = 0) -> JoypadSpace:
    """
    Create a Joypad-wrapped TetrisA-v3 env.
    • `skip`      – how many NES frames one call to env.step() will advance
    • `delay_ms`  – optional stagger delay to avoid nes-py DLL race
    """
    if delay_ms:
        time.sleep(delay_ms / 1000.0)

    core = gym_tetris.make("TetrisA-v3")     # nes-py env, no skip kwarg
    core = FrameSkip(core, k=skip)           # our custom frame-skipper
    return JoypadSpace(core, SIMPLE_MOVEMENT)


def reset_with_seed(env: JoypadSpace, seed: int | None = None) -> Tuple[np.ndarray, dict]:
    """
    Reseed env (old API) and return (obs, info).
    """
    if seed is not None:
        try:
            env.seed(seed)          # JoypadSpace 3.0.4 implements this
        except TypeError:
            env.unwrapped.seed(seed)
    obs = env.reset()
    # prime info dict (nes-py only populates on first step)
    obs, _, done, info = env.step(0)
    return obs, info

# ────────────────────────────── state helpers ─────────────────────────────────
PIECE_TO_IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'T':5,'Z':6}
N_ACTIONS    = len(SIMPLE_MOVEMENT)

def _column_heights(board20x10: np.ndarray) -> np.ndarray:
    h = 20 - board20x10[::-1].argmax(axis=0)
    h = np.where(board20x10.sum(axis=0)==0, 0, h)
    return (h // 4).astype(np.int8)          # 0–4 bins

def state_from_info(env: JoypadSpace, info: Dict) -> Tuple[int, ...]:
    """
    Build (piece_id, col-height bins) even if 'current_piece' is missing/None.
    """
    piece_raw = info.get("current_piece") or info.get("next_piece") or "I"
    piece_id  = PIECE_TO_IDX[piece_raw[0]]        # safe – always a char

    board = _get_board(env, info)                 # ← already robust
    return (piece_id, *tuple(_column_heights(board)))

# ────────────────────────────── reward shaping ────────────────────────────────
def shaped_reward(prev_info: dict, curr_info: dict, done: bool) -> float:
    lines = curr_info["number_of_lines"] - prev_info["number_of_lines"]
    agg_h = curr_info["board_height"]
    holes = curr_info.get("holes", 0)
    r = 1.0*lines - 0.1*holes - 0.01*agg_h
    if done:
        r -= 5.0
    return r

# ────────────────────────────── board extraction ──────────────────────────────
def _get_board(env: JoypadSpace, info: Dict) -> np.ndarray:
    """
    Return the current 20×10 board as a NumPy array, compatible with every
    gym-tetris release (0.7.x, 3.x, etc.).
    """
    # 1️⃣  Newest builds expose a helper
    if hasattr(env.unwrapped, "get_board"):
        return env.unwrapped.get_board()

    # 2️⃣  0.7.x kept it public
    if hasattr(env.unwrapped, "board"):
        return env.unwrapped.board

    # 3️⃣  3.0.x keeps it as _board  ← NEW branch
    if hasattr(env.unwrapped, "_board"):
        board = env.unwrapped._board
        return np.asarray(board).reshape(20, 10)

    # 4️⃣  Fallback: sometimes flattened in info
    if "board" in info:
        return np.asarray(info["board"], dtype=np.uint8).reshape(20, 10)

    raise AttributeError("Cannot extract board from this TetrisEnv build.")