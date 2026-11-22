"""The fast stages of the hierarchical decoder, and the features the router sees.

Pipeline for one shot (one memory basis), all stages loss-aware:

  1. BP        min-sum belief propagation, kit settings (80 iterations).
               If it converges (its answer reproduces the syndrome), accept it.
  2. OSD-0     if BP did not converge, the cheapest ordered-statistics step turns BP's soft
               output into an answer that always reproduces the syndrome. Whether that answer
               is the *right* one is not known, so:
  3. router    a learned classifier looks at features of the shot and of BP/OSD-0's output
               and predicts whether the OSD-0 answer is wrong. If it says "likely wrong",
  4. relay-BP  decodes the shot from scratch (the strongest, slowest decoder).

BP and OSD-0 run as one ldpc BpOsdDecoder call, so BP is not repeated. A separate BpDecoder
is timed only to know what BP alone costs on shots where it fails (needed to time the
"BP, then relay-BP" variant without a router).
"""

import pathlib
import sys
import time

import numpy as np
from ldpc import BpDecoder

KIT = pathlib.Path(__file__).resolve().parent.parent / "MidQWest-2026" / "challenge-1-ml-decoder-kit"
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from decoders import BPOSD_SETTINGS, bposd_decoder, relay_decoder  # noqa: E402

# Features the router sees. All are cheap to compute from the syndrome, the heralds and
# the BP/OSD-0 output, and every one has a plain meaning (useful when explaining the model).
FEATURES = [
    "syndrome_weight",       # number of detection events: more events = harder shot
    "heralds_fired",         # number of losses in this shot
    "bp_unsatisfied",        # checks BP's own guess still gets wrong (how far BP was from converging)
    "osd_weight",            # number of error mechanisms in the OSD-0 answer
    "osd_cost",              # sum of log((1-p)/p) over that answer: how unlikely it is a priori
    "bp_osd_disagree",       # mechanisms where BP's guess and the OSD-0 answer differ
    "osd_llr_support",       # sum of BP's posterior log-likelihood ratios over the OSD-0 answer
    "uncertain_bits",        # mechanisms BP is unsure about (|LLR| < 2)
    "heralded_in_answer",    # mechanisms in the answer that sit on an erased location
]


class FastStages:
    """Loss-aware BP + OSD-0 for one memory basis, with per-shot timing and router features."""

    def __init__(self, problem):
        self.problem = problem
        unaware = problem.unaware_priors()
        bp_settings = {k: BPOSD_SETTINGS[k] for k in ("max_iter", "bp_method", "ms_scaling_factor")}
        self.bposd0 = bposd_decoder(problem.H, unaware, osd_method="OSD_0", osd_order=0)
        self.bp = BpDecoder(problem.H, error_channel=list(unaware), **bp_settings)
        self.H = problem.H.tocsr()

    def run(self, syndrome, fired):
        """Decode one shot with BP, then OSD-0 if BP fails. Returns a dict of results."""
        start = time.perf_counter()
        priors = self.problem.aware_priors(fired)
        self.bposd0.update_channel_probs(priors)
        error = np.asarray(self.bposd0.decode(syndrome), np.uint8)
        fast_seconds = time.perf_counter() - start
        converged = bool(self.bposd0.converge)

        start = time.perf_counter()
        features = self.features(syndrome, fired, priors, error)
        feature_seconds = time.perf_counter() - start

        if converged:
            bp_seconds = fast_seconds
        else:  # what BP alone costs on this shot, for the variant that skips OSD-0
            start = time.perf_counter()
            self.bp.update_channel_probs(priors)
            self.bp.decode(syndrome)
            bp_seconds = time.perf_counter() - start

        return {"error": error, "converged": converged, "features": features,
                "fast_seconds": fast_seconds, "bp_seconds": bp_seconds, "feature_seconds": feature_seconds}

    def features(self, syndrome, fired, priors, error):
        d = self.bposd0
        bp_guess = np.asarray(d.bp_decoding, np.uint8)
        llr = np.asarray(d.log_prob_ratios)
        support = error.astype(bool)
        return np.array([
            syndrome.sum(),
            len(fired),
            ((self.H @ bp_guess + syndrome) % 2).sum(),
            support.sum(),
            np.log((1 - priors[support]) / priors[support]).sum(),
            (bp_guess != error).sum(),
            llr[support].sum(),
            (np.abs(llr) < 2).sum(),
            (priors[support] > 0.4).sum(),
        ], dtype=np.float64)


def timed_relay(problem, syndrome, fired):
    """Loss-aware relay-BP on one shot: (error, seconds). Rebuilt per shot (no prior-update call)."""
    start = time.perf_counter()
    error = relay_decoder(problem.H, problem.aware_priors(fired)).decode(syndrome)
    return np.asarray(error, np.uint8), time.perf_counter() - start


class TimedBpOsd:
    """Loss-aware BP-OSD at the kit's settings (order-10 combination sweep)."""

    def __init__(self, problem):
        self.problem = problem
        self.decoder = bposd_decoder(problem.H, problem.unaware_priors())

    def __call__(self, syndrome, fired):
        start = time.perf_counter()
        self.decoder.update_channel_probs(self.problem.aware_priors(fired))
        error = np.asarray(self.decoder.decode(syndrome), np.uint8)
        return error, time.perf_counter() - start
