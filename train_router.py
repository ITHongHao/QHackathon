"""Train the router on shots where BP did not converge: predict whether OSD-0's answer is wrong.

  python train_router.py --budget 0.001

Reads results/router_train_p*_loss{ratio}.npz. All error rates are pooled, so one router serves
every p (the features themselves tell it how noisy a shot is). 80% of the pairs train the models;
the other 20% (validation) set the threshold, so the test set is never used for any choice.

Threshold rule: accept the OSD-0 answer only while the accepted-but-wrong shots stay at most
`budget` of ALL basis-shots on validation (default 0.1%). This caps the accuracy the fast path
can cost; everything above the threshold is escalated to relay-BP.

Saves results/router_logistic.npz and results/router_gbm.pkl.
"""

import argparse
import glob
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from router import GbmRouter, LogisticRouter, choose_threshold
from two_stage import FEATURES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_split(split, loss_ratio):
    """Pool every error rate. Returns per basis-shot arrays plus each row's pair index."""
    X, wrong, converged, pair, ps = [], [], [], [], []
    offset = 0
    for path in sorted(glob.glob(os.path.join(RESULTS, f"router_{split}_p*_loss{loss_ratio:g}.npz"))):
        with np.load(path) as f:
            n = f["converged"].shape[0]
            X.append(f["features"].reshape(-1, len(FEATURES)))     # rows: pair 0 X, pair 0 Z, pair 1 X, ...
            wrong.append(f["fast_wrong"].reshape(-1))
            converged.append(f["converged"].reshape(-1))
            pair.append(np.repeat(np.arange(n) + offset, 2))
            ps.append(np.full(2 * n, float(f["p"])))
            offset += n
    if not X:
        raise FileNotFoundError(f"no router_{split}_*.npz files in {RESULTS}; run make_router_data.py first")
    return tuple(np.concatenate(a) for a in (X, wrong, converged, pair, ps))


def single_shot_microseconds(router, X, repeats=2000):
    rows = X[np.arange(repeats) % len(X)]
    start = time.perf_counter()
    for x in rows:
        router.escalate(x)
    return (time.perf_counter() - start) / repeats * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss-ratio", type=float, default=0.5)
    parser.add_argument("--budget", type=float, default=0.001,
                        help="max accepted-but-wrong shots, as a fraction of all basis-shots (validation)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, wrong, converged, pair, ps = load_split("train", args.loss_ratio)
    rng = np.random.default_rng(args.seed)
    val_pairs = rng.random(pair.max() + 1) < 0.2
    is_val = val_pairs[pair]
    print(f"{len(X):,} basis-shots, BP converged on {converged.mean():.1%}; "
          f"converged but wrong: {wrong[converged].sum()} (accepted by every variant)")

    # The router only ever sees shots where BP did not converge.
    hard = ~converged
    train, val = hard & ~is_val, hard & is_val
    print(f"router data: {train.sum():,} train / {val.sum():,} validation shots; "
          f"OSD-0 wrong on {wrong[train].mean():.1%} of them")
    n_val_all = is_val.sum()  # every validation basis-shot, the budget's denominator

    for cls in (LogisticRouter, GbmRouter):
        router = cls.fit(X[train], wrong[train])
        s_val = router.scores(X[val])
        threshold = choose_threshold(s_val, wrong[val], n_val_all, args.budget)
        router = router.with_threshold(threshold)
        accepted = s_val <= threshold
        print(f"\n{router.kind}: validation ROC AUC {roc_auc_score(wrong[val], s_val):.3f}, "
              f"{router.n_parameters:,} parameters, {single_shot_microseconds(router, X[val]):.1f} us per shot")
        print(f"  threshold {threshold:.4f}: accepts {accepted.mean():.1%} of non-converged shots, "
              f"escalates {(~accepted).sum() / n_val_all:.1%} of all shots to relay-BP; "
              f"accepted-but-wrong {wrong[val][accepted].sum()} ({wrong[val][accepted].sum() / n_val_all:.2%} of all)")
        for p in np.unique(ps):
            m = ps[val] == p
            print(f"  p={p}: escalates {(~accepted[m]).sum() / (is_val & (ps == p)).sum():.1%}, "
                  f"accepted-but-wrong {wrong[val][m][accepted[m]].sum()}")
        if router.kind == "logistic":
            print("  weights (on standardised log-features; + means 'more likely wrong'):")
            for name, w in sorted(zip(FEATURES, router.weights), key=lambda t: -abs(t[1])):
                print(f"    {name:20s} {w:+.2f}")
        router.save(os.path.join(RESULTS, f"router_{router.kind}." + ("npz" if router.kind == "logistic" else "pkl")))
    print(f"\nsaved -> results/router_logistic.npz, results/router_gbm.pkl")


if __name__ == "__main__":
    main()
