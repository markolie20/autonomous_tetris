"""
Environment creation + helper functions for the Tetris RL project.
This version adds richer board features **and** a refined shaped‑reward
function with a bigger signal‑to‑noise ratio.
Compatible with gym‑tetris 3.0.4 (old Gym API).
"""
from __future__ import annotations
import time, numpy as np, gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from typing import Tuple, Dict

from .frame_skip import FrameSkip  # custom wrapper

# ─────────────────────────── env factory ────────────────────────────────

def make_env(skip: int = 8, delay_ms: int = 0) -> JoypadSpace:
    """Return a Joypad‑wrapped **TetrisA‑v3** env with optional frame‑skip."""
    if delay_ms:
        time.sleep(delay_ms / 1000.0)
    core = gym_tetris.make("TetrisA-v3")
    core = FrameSkip(core, k=skip)
    return JoypadSpace(core, SIMPLE_MOVEMENT)

# ─────────────────────────── reset helper ───────────────────────────────

def reset_with_seed(env: JoypadSpace, seed: int | None = None) -> Tuple[np.ndarray, dict]:
    if seed is not None:
        try:
            env.seed(seed)
        except TypeError:            # older nes‑py
            env.unwrapped.seed(seed)
    obs = env.reset()
    obs, _, _, info = env.step(0)    # prime info
    return obs, info

# ─────────────────────────── state building ─────────────────────────────

PIECE_TO_IDX = {'I':0,'J':1,'L':2,'O':3,'S':4,'T':5,'Z':6}
N_ACTIONS    = len(SIMPLE_MOVEMENT)

# -- low‑level board analytics ------------------------------------------

def _column_heights(board: np.ndarray) -> np.ndarray:
    h = 20 - board[::-1].argmax(axis=0)
    h = np.where(board.sum(axis=0)==0, 0, h)
    return (h // 4).astype(np.int8)  # 5 buckets 0–4

def _aggregate_height(h: np.ndarray) -> int:
    return int(h.sum())

def _holes(board: np.ndarray) -> int:
    holes = 0
    for col in range(10):
        column = board[:, col]
        top = np.argmax(column[::-1] > 0)
        if top:
            holes += int((column[:20-top] == 0).sum())
    return holes

def _bumpiness(h: np.ndarray) -> int:
    return int(np.abs(np.diff(h)).sum())

def _max_well_depth(h: np.ndarray) -> int:
    depth = 0
    for i in range(10):
        left  = h[i-1] if i > 0 else 20
        right = h[i+1] if i < 9 else 20
        depth = max(depth, max(left, right) - h[i])
    return depth

# -- discretisers --------------------------------------------------------

def _bucket(x: int, bins) -> int:
    return int(np.digitize([x], bins, right=False)[0])

BINS_AH = [6, 12, 18]   # aggregate height 0‑3
BINS_HO = [0, 2, 5]     # holes            0‑3
BINS_BU = [4, 8, 12]    # bumpiness        0‑3
BINS_WE = [1, 3, 5]     # max well depth   0‑3

# -- public: convert env → state tuple ----------------------------------

def state_from_info(env: JoypadSpace, info: Dict) -> Tuple[int, ...]:
    piece_raw = info.get("current_piece") or info.get("next_piece") or "I"
    piece_id  = PIECE_TO_IDX[piece_raw[0]]

    board = _get_board(env, info)
    h     = _column_heights(board)

    return (
        piece_id,
        _bucket(_aggregate_height(h), BINS_AH),
        _bucket(_holes(board),        BINS_HO),
        _bucket(_bumpiness(h),        BINS_BU),
        _bucket(_max_well_depth(h),   BINS_WE),
        *tuple(h)
    )

# ─────────────────────────── reward shaping ─────────────────────────────

# Tuneable weights / constants
_LIVING_PENALTY = 0.002   # applied every frame
_HOLE_W          = 0.40
_HEIGHT_W        = 0.02
_TETRIS_BONUS    = 4.0    # extra for clearing exactly 4 lines


def shaped_reward(prev_info: dict, curr_info: dict, done: bool) -> float:
    """Dense reward: encourage line clears, punish structural flaws.

    * **Lines**: +1 per cleared line.
    * **Tetris bonus**: +4 when 4 lines cleared at once (makes Tetrises worth 8).
    * **Holes**: −0.4 × holes (big penalty).
    * **Aggregate height**: −0.02 × Σ column heights.
    * **Living**: −0.002 every frame (encourages faster play).
    * **Game‑over**: −5 flat penalty when the episode ends.
    """
    lines = curr_info["number_of_lines"] - prev_info["number_of_lines"]
    agg_h = curr_info["board_height"]
    holes = curr_info.get("holes", 0)

    r = (1.0 * lines
         + ( _TETRIS_BONUS if lines == 4 else 0 )
         - _HOLE_W * holes
         - _HEIGHT_W * agg_h
         - _LIVING_PENALTY)

    if done:
        r -= 5.0
    return r

# ─────────────────────────── board extraction ───────────────────────────

def _get_board(env: JoypadSpace, info: Dict) -> np.ndarray:
    if hasattr(env.unwrapped, "get_board"):
        return env.unwrapped.get_board()
    if hasattr(env.unwrapped, "board"):
        return env.unwrapped.board
    if hasattr(env.unwrapped, "_board"):
        return np.asarray(env.unwrapped._board).reshape(20, 10)
    if "board" in info:
        return np.asarray(info["board"], dtype=np.uint8).reshape(20, 10)
    raise AttributeError("Cannot extract board from this TetrisEnv build.")
