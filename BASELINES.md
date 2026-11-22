# Baselines

Every baseline measured so far for Challenge 1 (ML-assisted decoding of the [[144,12,12]] Gross
code).

## Conventions

These follow the challenge kit, so numbers can be compared with its reference table.

- **Shot pair.** Shot *k* of the X-memory experiment plus shot *k* of the Z-memory experiment.
  The pair **fails** if the decoder gets any of the 12 logical qubits wrong in either shot.
- **Logical error rate per cycle.** With failure fraction *P* over the pairs, the rate is
  1 - (1 - *P*)^(1/12). The ± value is one standard error.
- **Failures X, Z.** Failed shots in each memory experiment.
- **Latency.** Time to decode one pair (X shot + Z shot), one CPU core, Python.
  Measured on an Apple M5 with 4 worker processes, one per performance core.
- **Kit version.** Commit `31ceda8` of github.com/Infleqtion/MidQWest-2026, unmodified.
  Decoder settings are the kit's (the relay-BP authors' published settings).

---

## 1. No loss: the kit's baselines

This is the challenge's required reproduction step: 10,000 shot pairs per error rate, seed 1,
BP-OSD and relay-BP decoding the same shots. Per-shot results are in
`results/baselines_p*_n10000_seed1.npz`, the log in `results/baselines_sweep.log`.

| p | decoder | failures X, Z | rate per cycle (ours) | kit's reference rate | relay-BP authors' rate |
|---|---|---|---|---|---|
| 0.003 | BP-OSD | 36, 23 | 4.9e-4 ± 0.6e-4 | 4.7e-4 ± 0.6e-4 | 5.0e-4 |
| 0.003 | relay-BP | 5, 3 | 6.7e-5 ± 2.4e-5 | 4.2e-5 ± 1.9e-5 (8.0e-5 ± 0.8e-5 with 100k shots) | 6.0e-5 |
| 0.004 | BP-OSD | 266, 267 | 4.5e-3 ± 0.2e-3 | 4.4e-3 ± 0.2e-3 | 4.3e-3 |
| 0.004 | relay-BP | 59, 41 | 8.4e-4 ± 0.8e-4 | 8.7e-4 ± 0.9e-4 | 8.1e-4 |
| 0.005 | BP-OSD | 1310, 1344 | 2.36e-2 ± 0.05e-2 | 2.3e-2 ± 0.05e-2 | 2.3e-2 |
| 0.005 | relay-BP | 429, 431 | 7.3e-3 ± 0.3e-3 | 6.9e-3 ± 0.2e-3 | 7.1e-3 |

**Every number matches the reference within statistical error** (the largest gap, relay-BP at
p = 0.005, is 1.3 standard errors). The baselines are reproduced.

### Latency (ms per pair)

| p | decoder | mean | median | 99th percentile | max | kit's mean |
|---|---|---|---|---|---|---|
| 0.003 | BP-OSD | 60 | 10 | 386 | 678 | 140 |
| 0.003 | relay-BP | 46 | 29 | 315 | 3,094 | 100 |
| 0.004 | BP-OSD | 211 | 222 | 650 | 876 | 330 |
| 0.004 | relay-BP | 320 | 107 | 5,272 | 7,786 | 380 |
| 0.005 | BP-OSD | 424 | 412 | 876 | 1,224 | 590 |
| 0.005 | relay-BP | 1,098 | 305 | 6,295 | 10,838 | 1,700 |

- **Relay-BP's latency is heavy-tailed.** At p = 0.004 its median is 107 ms, but its slowest
  1% of pairs take over 5 s. At p = 0.005 the slowest 1% take over 6 s and the slowest pair
  almost 11 s. This tail is the main target of the two-stage decoder.
- **BP-OSD's median at p = 0.003 is 10 ms.** Most shots there are easy enough for BP to finish
  without needing OSD.
- **This machine is 1.2-2.3× faster than the kit authors'**, depending on decoder and p. That's
  less than the 2.3× suggested by the earlier 500-shot test. Quote your own timings, not theirs.

---

## 2. With qubit loss

**Loss model** (`loss.py`): after every two-qubit gate, each of its qubits is lost with
probability *p_loss* and replaced in a random state, and a flag records the loss. Here
*p_loss* = *p*/2, a placeholder chosen for testing and not taken from hardware data. That is
about 41 losses per shot at p = 0.004.

- **"ignores flags"**: the decoder knows the loss rate but not where losses happened.
- **"uses flags"**: each shot's flags raise the error probability at the lost qubits to 1/2.
  The fair baseline for any decoder that sees the flags.

### Smoke test: p = 0.004, p_loss = 0.002, 100 shot pairs, one core

From `loss_baselines.py --p 0.004 --loss 0.002 --shots 100` (seed 11).

| decoder | failures X, Z | rate per cycle | mean ms | 99th pct ms |
|---|---|---|---|---|
| BP-OSD, ignores flags | 38, 35 | 7.0e-2 ± 1.0e-2 | 723 | 1,185 |
| BP-OSD, uses flags | 5, 6 | 9.7e-3 ± 2.9e-3 | 432 | 908 |
| relay-BP, ignores flags | 17, 18 | 3.0e-2 ± 0.6e-2 | 3,660 | 11,667 |
| relay-BP, uses flags | 1, 1 | 1.7e-3 ± 1.2e-3 | 579 | 3,662 |

- **The flags are worth about 7× (BP-OSD) and 18× (relay-BP) in error rate**, before any ML,
  and they also make decoding faster.
- **Only 100 pairs:** the error bars are wide. A full test set of 2,000 pairs at
  p = 0.003, 0.004 and 0.005 comes next (`make_router_data.py --split test`). It runs the
  flag-aware decoders only.
- **These timings ran alongside the baseline sweep**, on the slower cores, so treat them as rough.

---

## 3. Fast-stage measurements

These are not baselines themselves, but they set up the two-stage decoder. X memory only, at
p = 0.004 and p_loss = 0.002, single core.

| stage | shots | how often it settles the shot | wrong when it does | time per shot |
|---|---|---|---|---|
| BP (80 iterations), converged | 60 | 50% | 0 of 30 | 16 ms mean over all 60 shots, converged or not |
| BP (80 iterations), converged | 120 | 52% | 0 of 63 | about 10 ms on the converged shots |
| OSD-0, after BP fails | 120 | the other 48% | 10 of 57 (18%) | about 28 ms in total |
| OSD order 10 (kit's BP-OSD), after BP fails | 120 | the other 48% | 3 of 57 (5%) | about 320 ms in total |

---

## 4. Planned no-ML controls (results pending)

The router has to beat these to show that the ML is doing the work. They are computed by
`evaluate_router.py` on the 2,000-pair test set.

| control | what it does |
|---|---|
| BP > relay-BP | accept BP when it converges, otherwise run relay-BP |
| BP > OSD-0 | always accept the fast answer, never run relay-BP |
| relay-BP alone | flag-aware relay-BP on every shot |
| BP-OSD alone | flag-aware BP-OSD at the kit's settings on every shot |

---

## Caveats

- **Relay-BP at p = 0.003 has only 8 failed pairs out of 10,000.** Its rate is uncertain by
  about ±35%. The kit authors used 100,000 shots for this point.
- **The no-loss latencies may be slightly inflated in places.** During the p = 0.004 and
  p = 0.005 runs, other work ran on the machine: the loss smoke test (9 min), the tests, the
  router dry runs, and for the last ~20 minutes of relay-BP at p = 0.005 the router training
  data (3 processes). That work probably landed on the efficiency cores, but the sweep's
  timings can't be guaranteed clean.
- **The loss model is optimistic** (a lost qubit is replaced at once, and loss happens only
  at gates), and *p_loss* = *p*/2 is not based on hardware data.

## Reproduce

```bash
cd gross_decoder
../.venv-kit/bin/python run_baselines.py --p 0.003 0.004 0.005 --shots 10000 --workers 4   # section 1, ~1.5 h
../.venv-kit/bin/python loss_baselines.py --p 0.004 --loss 0.002 --shots 100               # section 2, ~10 min
```
