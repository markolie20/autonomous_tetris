"""
Watch a trained tabular Q-learning agent play NES Tetris.

Usage (from project root)
-------------------------
# 1. Just show a window for one episode
python -m tetris_rl.play --model q_fast_decay

# 2. Record a MP4 in ./results/recordings/
python -m tetris_rl.play --model q_low_lr --record

Options
-------
--episodes N     how many episodes to watch      (default 1)
--skip     K     frame-skip for playback speed   (default 1 → full FPS)
--record        ️save gym Monitor MP4 to results/recordings/
"""
from __future__ import annotations
import argparse, os, time, pickle, numpy as np, gym
from collections import defaultdict
from pathlib import Path
from . import env_utils as eu, agent as ag, config as C

MODELS_DIR   = Path("results/models")
RECORD_DIR   = Path("results/recordings")
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── helpers ────────────────────────────────────
def load_agent(model_path: Path, rng) -> ag.QLearningAgent:
    """Reconstruct agent from a pickle and set ε = 0 (fully greedy)."""
    with model_path.open("rb") as f:
        payload = pickle.load(f)

    hp      = payload["hp"]
    q_table = payload["Q"]

    # create dummy agent, then overwrite its table
    agent = ag.QLearningAgent(rng, **hp)
    agent.Q = defaultdict(lambda: np.zeros(eu.N_ACTIONS, dtype=np.float32))
    for k, v in q_table.items():
        agent.Q[k] = v
    agent.eps = 0.0
    return agent


def run_episode(env: gym.Env, agent: ag.QLearningAgent, render: bool = True):
    _, info = eu.reset_with_seed(env)
    state   = eu.state_from_info(env, info)
    done    = False
    G       = 0.0

    while not done:
        if render:
            env.render()
            time.sleep(0.016)          # ~60 FPS

        a = int(np.argmax(agent.Q[state]))
        _, _, done, info = env.step(a)
        G += eu.shaped_reward(info, info, done)   # dummy prev=curr (only final)

        if not done:
            state = eu.state_from_info(env, info)

    return G


# ─────────────────────────── CLI ────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="model name without _model.pkl (e.g. q_fast_decay)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--skip", type=int, default=1,
                   help="frame-skip for playback (1 = realtime)")
    p.add_argument("--record", action="store_true",
                   help="wrap env in gym.Monitor and save MP4")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = MODELS_DIR / f"{args.model}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_path}")

    rng   = np.random.default_rng(C.SEED + 42)
    agent = load_agent(model_path, rng)

    # build env (skip can be 1 so we see every frame)
    env = eu.make_env(skip=args.skip)

    # optional Monitor wrapper
    if args.record:
        video_out = str(RECORD_DIR / f"{args.model}_{int(time.time())}")
        env = gym.wrappers.Monitor(env, video_out,
                                   force=True, video_callable=lambda ep: True)
        print(f"🎥  Recording to {video_out}.mp4")

    total = 0.0
    for ep in range(1, args.episodes + 1):
        ret = run_episode(env, agent, render=not args.record)
        total += ret
        print(f"Episode {ep}/{args.episodes}  |  return = {ret:7.2f}")

    print(f"Average return: {total / args.episodes:7.2f}")
    env.close()


if __name__ == "__main__":
    main()
