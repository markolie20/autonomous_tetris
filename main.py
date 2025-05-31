import os, warnings, gym, numpy as np, time
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(
    filename="run_log.txt",
    filemode="a", 
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
def log(msg):
    logging.info(msg)

# silence Gym warnings
os.environ["GYM_ENV_CHECKER"]      = "disabled"
os.environ["GYM_DISABLE_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
gym.logger.set_level(gym.logger.ERROR)

from tetris_rl import baseline, visualize, config as C

def _run_variant(
    name: str,
    hp: dict,
    seed_offset: int,
    delay_ms: int = 0,          
    skip: int = 12,              
):
    """
    Run one Q-learning variant in its own process.

    Parameters
    ----------
    name : str
        Identifier for the run (e.g. "q_fast_decay").
    hp : dict
        Hyper-parameter bundle forwarded to QLearningAgent.
    seed_offset : int
        Added to C.SEED so every worker has a unique RNG stream.
    delay_ms : int, optional
        Milliseconds to sleep before env creation (prevents DLL race).
    skip : int, optional
        Frame-skip factor passed to env_utils.make_env().

    Returns
    -------
    (name, returns ndarray, elapsed_seconds)
    """
    import time
    import numpy as np
    from tetris_rl import env_utils as eu, train, config as C

    start = time.perf_counter()

    env = eu.make_env(skip=skip, delay_ms=delay_ms)

    rng = np.random.default_rng(C.SEED + seed_offset)

    returns = train.train_variant(name, hp, rng, env)

    env.close()
    elapsed = time.perf_counter() - start
    log(f"Variant {name} finished in {elapsed:.1f}s (mean={np.mean(returns):.2f})")
    return name, returns, elapsed

def main():
    grand_start = time.perf_counter()

    log("Running random-policy baseline …")
    t0 = time.perf_counter()
    base_returns = baseline.run()
    baseline_time = time.perf_counter() - t0
    base_mean = base_returns.mean()
    log(f"Baseline done in {baseline_time:5.1f}s  |  µ = {base_mean:.2f}")

    curves, variant_times = {}, {}
    items       = list(C.VARIANTS.items())
    max_workers = min(cpu_count(), len(items))
    log(f"Launching {len(items)} variants on {max_workers} worker processes")

    tasks = [
        (vname, hp, i * 10_000, i * 250) 
        for i, (vname, hp) in enumerate(C.VARIANTS.items())
    ]

    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
        for name, rets, elapsed in pool.starmap(_run_variant, tasks):
            curves[name] = rets
            variant_times[name] = elapsed
            log(f"{name:<15s} finished in {elapsed:5.1f}s (µ={np.mean(rets):7.2f})")

    visualize.save_combined(base_mean, curves)
    visualize.save_per_variant(base_mean, curves)
    visualize.save_metrics(base_mean, curves, C.VARIANTS)
    visualize.save_smoothed_plots(curves, win=50)

    total_time = time.perf_counter() - grand_start
    log("All artefacts saved to ./results/")
    log("Timing summary:")
    log(f"   • baseline        : {baseline_time:6.1f} s")
    for n, t in variant_times.items():
        log(f"   • {n:<15s}: {t:6.1f} s")
    log(f"   • total runtime   : {total_time:6.1f} s")

if __name__ == "__main__":
    main()
