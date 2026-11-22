# ML-assisted decoding of the Gross code under qubit loss

Work for Challenge 1 of the Infleqtion "Qubits & Gradients" hackathon
(github.com/Infleqtion/MidQWest-2026). It builds on the challenge's decoder kit, cloned
next to this folder at commit `31ceda8`. The kit itself is not modified.

## Setup

```bash
brew install rust                                    # relay-bp builds from source
uv venv --python 3.12 .venv-kit                      # the kit is tested on Python 3.12
VIRTUAL_ENV=.venv-kit uv pip install -r MidQWest-2026/challenge-1-ml-decoder-kit/requirements.txt pytest
```

Run everything below from this folder with `../.venv-kit/bin/python`.

## Files

| file | what it does |
|---|---|
| `run_baselines.py` | the kit's BP-OSD and relay-BP baselines, saving per-shot failures and times |
| `loss.py` | adds heralded qubit loss to the kit's circuits; builds loss-aware decoder priors |
| `loss_baselines.py` | BP-OSD and relay-BP on lossy circuits, with and without the loss flags |
| `test_loss.py` | tests for the loss model (`python -m pytest test_loss.py`) |
| `results/` | saved baseline results (`.npz`) and logs |

## Loss model

**What is simulated.** After every two-qubit gate, each of its two qubits is lost with
probability `p_loss`. A lost qubit is immediately replaced by one in a random state (Stim's
`HERALDED_ERASE`: I, X, Y or Z with probability 1/4 each). A *herald* bit records the loss.
That is the loss-detection flag the challenge refers to. The Gross circuit has 20,736 such
gate-qubit slots per shot, so `p_loss = 0.002` gives about 41 losses per shot.

**How the decoders use it.** Each herald is a Stim detector, so the error model knows
which errors each loss can cause. A decoder that **ignores the flags** knows the loss rate
but not where losses happened. A decoder that **uses the flags** gets, for every shot, the
list of fired heralds. It sets the prior of each erased location to 1/2: in one memory basis,
two of the four random Paulis are visible.
- **BP-OSD** updates its priors in place.
- **Relay-BP** has no such call, so it is rebuilt for every shot.

Both costs count as decode time.

**Simplifications** (state these in the presentation):
- **Immediate replacement.** A real lost atom can stay missing for several gates before
  it is detected and replaced, which spreads errors further. The challenge recommends
  clifft for that more realistic model.
- **Where loss happens.** Loss occurs only on gate qubits, not during idling, preparation
  or measurement.
- **Loss rate.** `p_loss` is a free parameter. The results below use `p_loss = p / 2`,
  chosen for testing and not taken from hardware data.

## Results so far

**No loss**, 10,000 shot pairs per error rate: reproduced, matching the kit's reference
within statistical error. All baseline numbers are in [BASELINES.md](BASELINES.md).

**With loss**, smoke test of 100 shot pairs at p = 0.004 and p_loss = 0.002, single core.
The rate is per syndrome cycle; "failures" counts X and Z memory shots.

| decoder | failures X, Z | rate per cycle | mean / slowest 1% |
|---|---|---|---|
| BP-OSD, ignores flags | 38, 35 | 7.0e-2 ± 1.0e-2 | 723 / 1,185 ms |
| BP-OSD, uses flags | 5, 6 | 9.7e-3 ± 2.9e-3 | 432 / 908 ms |
| relay-BP, ignores flags | 17, 18 | 3.0e-2 ± 0.6e-2 | 3,660 / 11,667 ms |
| relay-BP, uses flags | 1, 1 | 1.7e-3 ± 1.2e-3 | 579 / 3,662 ms |

For comparison, relay-BP **without loss** at p = 0.004 has a rate of about 8.7e-4.

**What this means for the ML decoder.** Using the flags gives an 18× lower error rate
for relay-BP even without any ML, and it makes decoding faster too. So **the fair
baseline for a decoder that sees loss flags is a baseline that also sees them.** Beating
the flag-blind decoders would only show that the flags are useful, not that ML helps.
