"""Evaluate the hierarchical decoder against the baselines on the test set.

  python evaluate_router.py

Reads results/router_test_p*_loss{ratio}.npz (per-shot results of every stage on the same
shots) and the trained routers. Every variant is scored on exactly the same shot pairs:

  relay-BP             loss-aware relay-BP on every shot (the strongest baseline)
  BP-OSD               loss-aware BP-OSD at the kit's settings
  BP > relay-BP        no ML: accept BP when it converges, otherwise run relay-BP
  BP > OSD-0           no ML: always accept the fast answer (fast, less accurate)
  router (logistic)    BP > OSD-0 > learned router > relay-BP for the shots it flags
  router (GBM)         the same with the tree-ensemble router

A shot's latency is the sum of the measured times of the stages it actually went through;
a pair's latency is its X shot plus its Z shot, as in the kit. Router calls are timed here,
one shot at a time.

Writes results/evaluation_loss{ratio}.csv and two plots.
"""

import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from router import GbmRouter, LogisticRouter  # noqa: E402
from two_stage import KIT  # noqa: E402,F401  (puts the kit on sys.path)

from decoders import per_cycle_rate  # noqa: E402  (kit)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CYCLES = 12


def fixed_variants(d):
    """Variants without a router: name -> (wrong, seconds, escalated), each (pairs, 2)."""
    conv = d["converged"]
    none = np.zeros_like(conv)
    return {
        "relay-BP": (d["relay_wrong"], d["relay_seconds"], ~none),
        "BP-OSD": (d["bposd_wrong"], d["bposd_seconds"], none),
        "BP > relay-BP": (np.where(conv, d["fast_wrong"], d["relay_wrong"]),
                          np.where(conv, d["bp_seconds"], d["bp_seconds"] + d["relay_seconds"]), ~conv),
        "BP > OSD-0": (d["fast_wrong"], d["fast_seconds"], none),
    }


def routed(d, escalate, router_seconds):
    """BP > OSD-0 > router > relay-BP, given per-shot escalation decisions and router times."""
    conv = d["converged"]
    esc = ~conv & escalate
    wrong = np.where(esc, d["relay_wrong"], d["fast_wrong"])
    seconds = d["fast_seconds"] + np.where(conv, 0.0, d["feature_seconds"] + router_seconds) \
        + np.where(esc, d["relay_seconds"], 0.0)
    return wrong, seconds, esc


def run_router(router, d):
    """Escalation decision and timed router call for every shot where BP did not converge."""
    F, conv = d["features"], d["converged"]
    escalate, seconds = np.zeros(conv.shape, bool), np.zeros(conv.shape)
    for i, j in zip(*np.nonzero(~conv)):
        start = time.perf_counter()
        escalate[i, j] = router.escalate(F[i, j])
        seconds[i, j] = time.perf_counter() - start
    return escalate, seconds


def summarize(name, p, wrong, seconds, escalated, reference_failed):
    failed = wrong.any(axis=1)
    rate, err = per_cycle_rate(failed, CYCLES)
    ms = seconds.sum(axis=1) * 1e3
    only_this, only_ref = int(np.sum(failed & ~reference_failed)), int(np.sum(~failed & reference_failed))
    pvalue = binomtest(only_this, only_this + only_ref).pvalue if only_this + only_ref else 1.0
    return {"p": p, "decoder": name, "pairs": len(failed),
            "failures_X": int(wrong[:, 0].sum()), "failures_Z": int(wrong[:, 1].sum()),
            "ler_per_cycle": rate, "ler_err": err,
            "mean_ms": ms.mean(), "p99_ms": np.percentile(ms, 99), "max_ms": ms.max(),
            "escalated_pct": 100 * escalated.mean(),
            "pairs_only_this_failed": only_this, "pairs_only_relay_failed": only_ref,
            "paired_pvalue_vs_relay": pvalue}


# Colorblind-validated categorical palette, fixed order; text stays in neutral ink.
COLORS = {"router (logistic)": "#2a78d6", "relay-BP": "#eb6834", "BP > relay-BP": "#1baf7a",
          "BP-OSD": "#eda100", "BP > OSD-0": "#e87ba4"}
MARKERS = {"router (logistic)": "o", "relay-BP": "s", "BP > relay-BP": "D", "BP-OSD": "^", "BP > OSD-0": "v"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_vs_p(df, path, loss_ratio):
    names = ["router (logistic)", "relay-BP", "BP > relay-BP", "BP-OSD"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, facecolor=SURFACE)
    panels = [("ler_per_cycle", "logical error rate per cycle", "Logical error rate"),
              ("mean_ms", "ms per shot pair", "Mean decoding latency"),
              ("p99_ms", "ms per shot pair", "99th-percentile latency")]
    for ax, (col, ylabel, title) in zip(axes, panels):
        for name in names:
            rows = df[df.decoder == name].sort_values("p")
            ax.errorbar(rows.p, rows[col], yerr=rows.ler_err if col == "ler_per_cycle" else None,
                        color=COLORS[name], marker=MARKERS[name], markersize=5, linewidth=1.5, capsize=2,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        ax.set_yscale("log")
        ax.set_xticks(sorted(df.p.unique()))
        style(ax, "physical error rate p", ylabel, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"Gross code [[144,12,12]], 12 cycles, loss = {loss_ratio:g} x p, all decoders use loss flags",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(curves, points, path):
    ps = sorted(curves)
    fig, axes = plt.subplots(1, len(ps), figsize=(4.4 * len(ps), 4.0), dpi=150, facecolor=SURFACE, squeeze=False)
    for ax, p in zip(axes[0], ps):
        mean_ms, ler = curves[p]
        ax.plot(mean_ms, ler, color=COLORS["router (logistic)"], linewidth=1.5)
        for name, (x, y) in points[p].items():
            ax.plot(x, y, marker=MARKERS[name], color=COLORS[name], markersize=7, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=1)
            ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "mean latency per pair (ms)", "logical error rate per cycle", f"p = {p}")
    fig.suptitle("Accuracy vs speed: the line sweeps the logistic router's threshold (lower-left is better)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss-ratio", type=float, default=0.5)
    args = parser.parse_args()

    routers = [LogisticRouter.load(os.path.join(RESULTS, "router_logistic.npz")),
               GbmRouter.load(os.path.join(RESULTS, "router_gbm.pkl"))]
    rows, curves, points, rss = [], {}, {}, []
    for path in sorted(glob.glob(os.path.join(RESULTS, f"router_test_p*_loss{args.loss_ratio:g}.npz"))):
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        p = float(d["p"])
        rss.append(float(d.get("worker_peak_rss_mb", np.nan)))
        variants = fixed_variants(d)
        for router in routers:
            escalate, router_seconds = run_router(router, d)
            variants[f"router ({router.kind})"] = routed(d, escalate, router_seconds)
        relay_failed = d["relay_wrong"].any(axis=1)
        rows += [summarize(name, p, *v, relay_failed) for name, v in variants.items()]

        # Threshold sweep for the logistic router (batch scores, mean router time).
        scores = routers[0].scores(d["features"].reshape(-1, d["features"].shape[-1])).reshape(d["converged"].shape)
        mean_router_s = np.mean(run_router(routers[0], d)[1][~d["converged"]])
        sweep = []
        for t in np.unique(np.quantile(scores[~d["converged"]], np.linspace(0, 1, 60))):
            w, s, _ = routed(d, scores > t, mean_router_s)
            sweep.append((s.sum(axis=1).mean() * 1e3, per_cycle_rate(w.any(axis=1), CYCLES)[0]))
        curves[p] = tuple(np.array(sweep).T)
        points[p] = {r["decoder"]: (r["mean_ms"], r["ler_per_cycle"]) for r in rows
                     if r["p"] == p and r["decoder"] in COLORS}

    if not rows:
        raise FileNotFoundError("no router_test_*.npz files; run make_router_data.py --split test first")
    df = pd.DataFrame(rows)
    csv = os.path.join(RESULTS, f"evaluation_loss{args.loss_ratio:g}.csv")
    df.to_csv(csv, index=False)
    plot_vs_p(df, os.path.join(RESULTS, f"ler_latency_loss{args.loss_ratio:g}.png"), args.loss_ratio)
    plot_tradeoff(curves, points, os.path.join(RESULTS, f"tradeoff_loss{args.loss_ratio:g}.png"))

    show = ["p", "decoder", "failures_X", "failures_Z", "ler_per_cycle", "ler_err", "mean_ms", "p99_ms",
            "escalated_pct", "paired_pvalue_vs_relay"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nresources (single CPU core per shot, no GPU):")
    for r in routers:
        print(f"  router ({r.kind}): {r.n_parameters:,} parameters")
    print(f"  peak memory of a decoding worker (circuit, error model, all decoders): {np.nanmax(rss):.0f} MB")
    print(f"\nsaved -> {os.path.relpath(csv)} and the plots next to it")


if __name__ == "__main__":
    main()
