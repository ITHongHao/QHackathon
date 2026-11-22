"""Run the kit's BP-OSD and relay-BP baselines and save per-shot results.

This does the same computation as the kit's baseline.py: same circuits, seeds,
decoder settings and worker split. It calls the kit's own functions. In addition
it saves, for every shot, whether each decoder failed and how long it took. That
lets a new decoder be compared on exactly the same shots later (a paired test)
without re-running these slow baselines.

  python run_baselines.py --p 0.003 0.004 0.005 --shots 10000 --workers 4

Writes results/baselines_p{p}_n{shots}_seed{seed}.npz per error rate.
"""

import argparse
import pathlib
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "MidQWest-2026" / "challenge-1-ml-decoder-kit"
sys.path.insert(0, str(KIT))  # import the kit's modules without copying them

from baseline import BASELINES, memory_experiments, score_slice, split_shots  # noqa: E402
from circuit import GROSS_CIRCUITS  # noqa: E402
from decoders import print_score  # noqa: E402

RESULTS = HERE / "results"


def unsplit(per_slice, pieces):
    """Undo split_shots: slice k holds shots k, k + pieces, ...; put them back in shot order."""
    n = sum(len(a) for a in per_slice)
    out = np.empty((n,) + per_slice[0].shape[1:], per_slice[0].dtype)
    for k, a in enumerate(per_slice):
        out[k::pieces] = a
    return out


def run(p, shots, seed, workers, cycles=12):
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], p, shots, seed)
    slices = split_shots(experiments, workers)
    print(f"{shots} shots at p={p}, {cycles} cycles, seed {seed}, {workers} workers, failures listed as [X Z]", flush=True)
    saved = {}
    with Pool(workers) as pool:
        for name in BASELINES:
            start = time.perf_counter()
            results = pool.map(score_slice, [(name, exps) for exps in slices])
            failed = unsplit([r[0] for r in results], workers)    # shots x [X, Z]
            seconds = unsplit([r[1] for r in results], workers)   # shots x [X, Z]
            print_score(name, failed, seconds, cycles)
            print(f"{'':10s} wall time {time.perf_counter() - start:.0f} s", flush=True)
            key = name.replace("-", "_")
            saved[f"{key}_failed"], saved[f"{key}_seconds"] = failed, seconds

    kit_commit = subprocess.run(["git", "-C", str(KIT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"baselines_p{p:.4f}_n{shots}_seed{seed}.npz"
    np.savez_compressed(path, p=p, shots=shots, seed=seed, workers=workers, cycles=cycles,
                        kit_commit=kit_commit, **saved)
    print(f"saved -> {path.relative_to(HERE)}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1, help="X memory uses seed, Z memory seed + 1 (kit convention)")
    # One worker per performance core keeps the latency numbers comparable:
    # more workers spill onto the slower efficiency cores and inflate per-shot times.
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for p in args.p:
        run(p, args.shots, args.seed, args.workers)


if __name__ == "__main__":
    main()
