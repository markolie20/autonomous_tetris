# Autonomous Tetris — tabular Q-learning showcase
A compact research playground that teaches a vanilla Q-learning agent to
survive and improve in **NES Tetris** (gym-tetris 3.0.4).  
You get a clean baseline, three hyper-parameter variants, parallel
training, automatic visualisations (raw & smoothed) and saved artefacts
(models | metrics | plots)

---

## 1. Project layout
```
autonomous_tetris/
│ main.py ← orchestrates baseline + parallel variants
│ requirements.txt
│ README.md
├── tetris_rl/ ← reusable package
│ ├── agent.py Q-learning agent (ε-greedy, tabular)
│ ├── baseline.py random-policy baseline
│ ├── config.py central hyper-parameters & magic numbers
│ ├── env_utils.py env factory + state & reward helpers
│ ├── frame_skip.py custom k-frame skip NES wrapper
│ ├── train.py per-variant training loop
│ └── visualize.py plots, CSV, JSON
│ └── play.py Seeing the models play
└── results/ ← auto-generated (see §5)

```
---

## 2. Key features

* **Frame-skip (8 frames)** — Each action is repeated for 8 game frames, greatly speeding up training (about 6× faster per episode) while maintaining playable game dynamics.
* **Multiprocessing** — one emulator per CPU core, staggered start to avoid
* **Baseline vs. variants**  
  - `q_fast_decay`  ε drops quickly  
  - `q_slow_decay`  gentler exploration schedule  
  - `q_low_lr`      half learning-rate
* **Once-per-frame ε-decay** — The exploration rate (epsilon) is decayed after every frame.
* **Reward shaping**  
  `+1·lines  –0.1·holes  –0.01·aggregate height  –5 on game-over`
* **Automatic artefacts**  
  CSV per episode, summary JSON, raw & smoothed plots, pickled Q-tables.

---

## 3. Installation

1.  Python ≥ 3.10
2.  `pip install -r requirements.txt`  
3.  On Windows make sure the **Microsoft C++ Build Tools** are present.

---

## 4. Running

*Standard run (baseline + 3 variants, 15k episodes each)*  
`python main.py`

Options are edited in `tetris_rl/config.py`:

| field                 | meaning                                  |
|-----------------------|------------------------------------------|
| `SEED`                | global RNG seed                          |
| `BASELINE_EPISODES`   | baseline length                          |
| `Q_LEARNING_EPISODES` | training episodes per variant            |
| `MAX_FRAMES`          | NES frames per episode (post skip)       |
| `VARIANTS`            | dict of hyper-parameter bundles          |
| `PRINT_EVERY_TRAIN`   | frequency of log lines                   |

---

## 5. Output files (`results/`)

| sub-folder | contents |
|------------|----------|
| `data/`    | `<variant>_episodes.csv` (episode, reward, advantage) and `summary.json` |
| `models/`  | pickled Q-tables `<variant>_model.pkl` |
| `plots/`   | raw curves (`combined.png`, `<variant>.png`) and smoothed versions (`*_smooth.png`) |

---

## 6. Understanding the code

* **`agent.py`**  
  Stores Q in a Python `defaultdict`; update is plain
  Bellman backup.  ε-decay happens *after* each episode once the global
  frame counter crosses `decay_after`.
* **`env_utils.py`**  
  Provides a robust way to fetch the 20×10 board across all gym-tetris
  versions.  The state fed to Q is  
  `(current piece id, 10 column-height bins)`.
* **`frame_skip.FrameSkip`**  
  Repeats the last action `k – 1` times and returns only the final
  observation; rewards from nes-py are ignored because we compute our own
  shaped reward.
* **`train.py`**  
  Accepts an existing env (for the multiprocessing workers) **or** builds
  one if run stand-alone.  Saves the Q-table at the end.
* **`visualize.py`**  
  Base plots + CSV/JSON **and** a helper that applies rolling mean
  (`win`) *or* EWMA (`α`) before plotting.
* **`play.py`**
  To run play.py you have to use this: `python -m tetris_rl.play --model q_low_lr --record`  
  This will open up a window where you can see the chosen model play the actual tetris game.
---

## 7. Extending the experiment

* **Add a new variant**  
  Append to `VARIANTS` in `config.py`; each entry gets its own process.
* **Longer training**  
  Increase `Q_LEARNING_EPISODES` and possibly slow `eps_decay`.
* **Alternative smoothing**  
  Change the call in `main.py`  
  `visualize.save_smoothed_plots(curves, ewma=0.1)` for exponential.
* **Different state features**  
  Tweak `state_from_info` in `env_utils.py` (e.g. add holes/bumpiness).
* **Switch to SARSA**  
  Replace `best_next` with the value of the actually sampled `a′` in
  `agent.update`.

---

## 8. Troubleshooting

* **Access-violation on Windows** → the staggered `delay_ms` already
  mitigates this.  Increase the step (e.g. 500 ms) if crashes persist.
* **Gym API warnings** – gym-tetris uses the old single-`done` signature;
  warnings are silenced at the top of `main.py`.
* **Extremely slow training** – make sure you have 8-frame skip and that
  all CPU cores are used (`task manager`).

---