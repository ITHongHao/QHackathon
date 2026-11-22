"""BP-OSD and relay-BP on the lossy Gross code, with and without the loss flags.

  python loss_baselines.py --p 0.004 --loss 0.002 --shots 100

Four decoders, all at the kit's settings and on the same shots:
  *-unaware  know the loss rate but not where losses happened (fixed priors, heralds ignored)
  *-aware    get each shot's herald flags and raise the priors of the erased locations.
             BP-OSD updates its priors in place; relay-BP has no such call, so it is rebuilt
             for every shot. Both costs are inside the timed region: they are part of decoding.

Single process, so the timings are per-shot latencies on one core.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

KIT = pathlib.Path(__file__).resolve().parent.parent / "MidQWest-2026" / "challenge-1-ml-decoder-kit"
sys.path.insert(0, str(KIT))

from circuit import GROSS_CIRCUITS, load_circuit  # noqa: E402
from decoders import bposd_decoder, print_score, relay_decoder  # noqa: E402

from loss import add_heralded_loss, loss_problem  # noqa: E402


def make_decoders(problem):
    """name -> function(syndrome, fired heralds) -> error vector over the columns of H."""
    unaware = problem.unaware_priors()
    bposd_fixed = bposd_decoder(problem.H, unaware)
    bposd_updated = bposd_decoder(problem.H, unaware)
    relay_fixed = relay_decoder(problem.H, unaware)

    def bposd_aware(syndrome, fired):
        bposd_updated.update_channel_probs(problem.aware_priors(fired))
        return bposd_updated.decode(syndrome)

    def relay_aware(syndrome, fired):
        return relay_decoder(problem.H, problem.aware_priors(fired)).decode(syndrome)

    return {
        "bposd-unaware": lambda s, f: bposd_fixed.decode(s),
        "bposd-aware": bposd_aware,
        "relay-unaware": lambda s, f: relay_fixed.decode(s),
        "relay-aware": relay_aware,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004, help="Pauli noise strength (the kit's p)")
    parser.add_argument("--loss", type=float, default=0.002, help="loss probability per qubit per two-qubit gate")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()

    failed, seconds = {}, {}
    for k, basis in enumerate("XZ"):
        start = time.perf_counter()
        circuit = add_heralded_loss(load_circuit(GROSS_CIRCUITS[basis], args.p), args.loss)
        problem = loss_problem(circuit, basis)
        print(f"{basis} memory: {problem.H.shape[0]} check detectors, {len(problem.herald_ids)} heralds, "
              f"{problem.H.shape[1]} columns, built in {time.perf_counter() - start:.1f} s")

        det, obs = circuit.compile_detector_sampler(seed=args.seed + k).sample(args.shots, separate_observables=True)
        syndromes = det[:, problem.det_ids].astype(np.uint8)
        heralds = det[:, problem.herald_ids]
        print(f"  heralds fired per shot: mean {heralds.sum(1).mean():.1f}, max {heralds.sum(1).max()}")

        for name, decode in make_decoders(problem).items():
            f, s = failed.setdefault(name, []), seconds.setdefault(name, [])
            f.append(np.zeros(args.shots, bool))
            s.append(np.zeros(args.shots))
            for shot in range(args.shots):
                fired = np.flatnonzero(heralds[shot])
                t0 = time.perf_counter()
                error = np.asarray(decode(syndromes[shot], fired), np.uint8)
                s[-1][shot] = time.perf_counter() - t0
                f[-1][shot] = ((problem.L @ error) % 2 != obs[shot]).any()

    print(f"\n{args.shots} shots at p={args.p}, loss={args.loss}, {args.cycles} cycles, failures listed as [X Z]")
    for name in failed:
        print_score(name, np.column_stack(failed[name]), np.column_stack(seconds[name]), args.cycles)


if __name__ == "__main__":
    main()
