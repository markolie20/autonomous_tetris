# tetris_rl/visualize.py  ─────────────────────────────────────────────────
import os, csv, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
RESULTS_DIR = "results"
PLOTS_DIR  = os.path.join(RESULTS_DIR, "plots_smooth")
DATA_DIR   = os.path.join(RESULTS_DIR, "data")

# ╭───────────────────────── helpers ─────────────────────────╮
def _ensure_dir(path=RESULTS_DIR):
    if not os.path.exists(path):
        os.makedirs(path)

# ╰───────────────────────────────────────────────────────────╯


# ────────────────────────── base plots (unchanged) ──────────────────────
def save_combined(baseline_mean: float, curves: dict[str, np.ndarray]):
    _ensure_dir()
    plt.figure(figsize=(9,5))
    x = np.arange(1, len(next(iter(curves.values())))+1)
    for name, ret in curves.items():
        plt.plot(x, ret - baseline_mean, label=name)
    plt.axhline(0, color='red', ls='--')
    plt.xlabel("Episode");  plt.ylabel("Reward − baseline µ")
    plt.title("All Q-learning variants vs baseline")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined.png"))
    plt.close()


def save_per_variant(baseline_mean: float, curves: dict[str, np.ndarray]):
    _ensure_dir()
    x = np.arange(1, len(next(iter(curves.values())))+1)
    for name, ret in curves.items():
        plt.figure(figsize=(9,5))
        plt.plot(x, ret - baseline_mean, label=name)
        plt.axhline(0, color='red', ls='--')
        plt.xlabel("Episode"); plt.ylabel("Reward − baseline µ")
        plt.title(f"{name} vs baseline")
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{name}.png"))
        plt.close()


# ────────────────────────── CSV & summary (unchanged) ───────────────────
def save_metrics(baseline_mean: float,
                 curves: dict[str, np.ndarray],
                 hp_grid : dict[str, dict]):
    _ensure_dir()
    summary = []

    for name, ret in curves.items():
        adv  = ret - baseline_mean
        csv_path = os.path.join(DATA_DIR, f"{name}_episodes.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "advantage"])
            for ep, (r, a) in enumerate(zip(ret, adv), start=1):
                w.writerow([ep, float(r), float(a)])

        summary.append({
            "variant"        : name,
            "hyperparameters": hp_grid[name],
            "episodes"       : len(ret),
            "mean_reward"    : float(np.mean(ret)),
            "mean_advantage" : float(np.mean(adv)),
            "final_reward"   : float(ret[-1]),
            "best_reward"    : float(np.max(ret)),
            "baseline_mean"  : float(baseline_mean),
        })

    with open(os.path.join(DATA_DIR, "summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2)


# ────────────────────────── NEW: smooth visualisation ────────────────────
def _smooth(series: pd.Series, win: int, ewma: float | None):
    if ewma is not None:
        return series.ewm(alpha=ewma, adjust=False).mean()
    return series.rolling(win, min_periods=1).mean()


def save_smoothed_plots(curves: dict[str, np.ndarray],
                        win : int = 50,
                        ewma: float | None = None):
    """
    Create noise-reduced plots for every variant **and** a combined plot.

    Parameters
    ----------
    curves : dict[str, np.ndarray]
        Raw per-episode reward curves.
    win    : int
        Rolling-mean window (ignored if `ewma` is given).
    ewma   : float | None
        Alpha for exponential smoothing.  Use None to disable.
    """
    _ensure_dir(PLOTS_DIR)

    combined_fig = plt.figure(figsize=(10,5))
    ax_comb = combined_fig.add_subplot(1,1,1)

    for name, rewards in curves.items():
        ep = np.arange(1, len(rewards)+1)
        df = pd.DataFrame({"episode": ep, "reward": rewards})
        df["smooth"] = _smooth(df["reward"], win, ewma)

        # combined
        ax_comb.plot(df["episode"], df["smooth"], label=name)

        # individual
        plt.figure(figsize=(8,4))
        plt.plot(df["episode"], df["reward"],  alpha=0.25, label="raw")
        plt.plot(df["episode"], df["smooth"], lw=2,      label="smoothed")
        plt.title(f"{name} – smoothed learning curve")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{name}_smooth.png"))
        plt.close()

    title = (f"Rolling mean (w={win})" if ewma is None
             else f"EWMA (α={ewma})")
    ax_comb.set_title(title)
    ax_comb.set_xlabel("Episode"); ax_comb.set_ylabel("Reward")
    ax_comb.legend(); ax_comb.grid(True)
    combined_fig.tight_layout()
    combined_fig.savefig(os.path.join(PLOTS_DIR, "combined_smooth.png"))
    plt.close(combined_fig)

    print(f"✅  Smoothed plots saved → {PLOTS_DIR}")
