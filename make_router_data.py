"""Generate router training data, and per-shot results of every decoder on the test set.

  python make_router_data.py --split train --shots 6000   # BP + OSD-0 only: cheap
  python make_router_data.py --split test  --shots 2000   # also relay-BP and BP-OSD: slow

For each error rate p (loss probability = p * loss_ratio), shots are cut into chunks and dealt
to worker processes. Each chunk samples its own X and Z memory shots with its own seed (train
and test seeds never overlap) and decodes them. Shot k of X memory and shot k of Z memory form
a pair, as in the kit.

Saves results/router_{split}_p{p}_loss{ratio}.npz. Every array has one row per pair and one
column per basis [X, Z] (features have a third axis):
  features          the router's inputs (two_stage.FEATURES)
  converged         BP converged
  fast_wrong        the BP / OSD-0 answer gets a logical observable wrong
  fast_seconds      time of BP, plus OSD-0 when BP fails
  bp_seconds        time of BP alone
  feature_seconds   time to compute the features
  relay_wrong / relay_seconds, bposd_wrong / bposd_seconds   (test split only)
"""

import argparse
import os
import resource
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from two_stage import FEATURES, KIT, FastStages, TimedBpOsd, timed_relay  # also puts the kit on sys.path

from circuit import GROSS_CIRCUITS, load_circuit  # noqa: E402  (kit)
from loss import add_heralded_loss, loss_problem  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED_BASE = {"train": 10_000, "test": 20_000}


def decode_chunk(job):
    """Sample and decode one chunk of shot pairs at one error rate. Runs in a worker process."""
    p, p_loss, chunk, shots, split = job
    columns = {}
    for b, basis in enumerate("XZ"):
        circuit = add_heralded_loss(load_circuit(GROSS_CIRCUITS[basis], p), p_loss)
        problem = loss_problem(circuit, basis)
        seed = SEED_BASE[split] + int(round(p * 1e5)) * 100 + chunk * 2 + b
        det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        syndromes = det[:, problem.det_ids].astype(np.uint8)
        heralds = det[:, problem.herald_ids]
        del det

        fast = FastStages(problem)
        bposd = TimedBpOsd(problem) if split == "test" else None
        rows = []
        for k in range(shots):
            fired = np.flatnonzero(heralds[k])
            r = fast.run(syndromes[k], fired)
            row = {
                "features": r["features"],
                "converged": r["converged"],
                "fast_wrong": wrong(problem, r["error"], obs[k]),
                "fast_seconds": r["fast_seconds"],
                "bp_seconds": r["bp_seconds"],
                "feature_seconds": r["feature_seconds"],
            }
            if split == "test":
                error, row["relay_seconds"] = timed_relay(problem, syndromes[k], fired)
                row["relay_wrong"] = wrong(problem, error, obs[k])
                error, row["bposd_seconds"] = bposd(syndromes[k], fired)
                row["bposd_wrong"] = wrong(problem, error, obs[k])
            rows.append(row)
        for key in rows[0]:
            columns.setdefault(key, []).append(np.array([row[key] for row in rows]))
    # Peak memory of this worker process so far (circuit, error model, decoders, shots).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3   # bytes on macOS, KiB on Linux
    # stack the two bases on axis 1: (shots, 2) or (shots, 2, features)
    return p, chunk, {key: np.stack(cols, axis=1) for key, cols in columns.items()}, rss_mb


def wrong(problem, error, observables):
    return bool(((problem.L @ error) % 2 != observables).any())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--loss-ratio", type=float, default=0.5, help="loss probability = p * loss_ratio")
    parser.add_argument("--shots", type=int, default=2000, help="shot pairs per error rate")
    parser.add_argument("--chunks", type=int, default=8, help="chunks per error rate (load balancing)")
    parser.add_argument("--workers", type=int, default=4, help="one per performance core keeps timings comparable")
    args = parser.parse_args()

    sizes = np.diff(np.linspace(0, args.shots, args.chunks + 1).astype(int))
    jobs = [(p, p * args.loss_ratio, c, int(n), args.split) for p in args.p for c, n in enumerate(sizes) if n > 0]
    # slowest error rate first so the long jobs don't trail at the end
    jobs.sort(key=lambda j: -j[0])

    print(f"{args.split}: {args.shots} pairs per p at p={args.p}, loss ratio {args.loss_ratio}, "
          f"{len(jobs)} chunks on {args.workers} workers", flush=True)
    start = time.perf_counter()
    done = {p: {} for p in args.p}
    peak_rss_mb = 0.0
    with Pool(args.workers) as pool:
        for p, chunk, arrays, rss_mb in pool.imap_unordered(decode_chunk, jobs):
            done[p][chunk] = arrays
            peak_rss_mb = max(peak_rss_mb, rss_mb)
            print(f"  p={p} chunk {chunk} done ({time.perf_counter() - start:.0f} s, worker peak {rss_mb:.0f} MB)", flush=True)

    kit_commit = subprocess.run(["git", "-C", str(KIT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(RESULTS, exist_ok=True)
    for p, chunks in done.items():
        merged = {key: np.concatenate([chunks[c][key] for c in sorted(chunks)]) for key in chunks[0]}
        path = os.path.join(RESULTS, f"router_{args.split}_p{p:.4f}_loss{args.loss_ratio:g}.npz")
        np.savez_compressed(path, p=p, p_loss=p * args.loss_ratio, feature_names=FEATURES,
                            kit_commit=kit_commit, worker_peak_rss_mb=peak_rss_mb, **merged)
        conv = merged["converged"]
        print(f"p={p}: {len(conv)} pairs  BP converged {conv.mean():.0%}  "
              f"BP/OSD-0 wrong when not converged {merged['fast_wrong'][~conv].mean():.1%}"
              + (f"  relay wrong {merged['relay_wrong'].mean():.2%}" if "relay_wrong" in merged else "")
              + f"  -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
