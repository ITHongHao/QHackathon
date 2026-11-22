"""Qubit loss for the kit's circuits, modelled as heralded erasure, plus loss-aware decoding inputs.

Loss model
----------
Neutral atoms can be lost from their traps. We use the standard first approximation,
"erasure with immediate replacement": after every two-qubit gate layer, each qubit that took
part in a gate is lost with probability p_loss. A lost qubit is replaced by one in a completely
random state (Stim's HERALDED_ERASE applies I, X, Y or Z with probability 1/4 each), and a
herald bit records that it happened. The herald bit is the "loss-detection flag" of the challenge.

This is optimistic. A real lost atom can stay missing for several gates before it is detected
and replaced, which spreads errors further. clifft (recommended by the challenge) can simulate
that. This model gets the decoding pipeline working first.

Each herald is declared as a DETECTOR with coordinates (2, herald index). That does two things:
  - sampled detector bits include the herald flags, so a decoder can use them;
  - Stim's detector error model ties every erasure error to its herald detector, which is how
    we find the error mechanisms that a fired herald makes likely.
The kit selects X and Z check detectors by coordinate (0,) and (1,), so its functions ignore heralds.
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import stim

HERALD = 2.0                    # first detector coordinate of a herald
CHECK = {"X": 0.0, "Z": 1.0}    # the kit's coordinates for X- and Z-check detectors
MIN_PRIOR = 1e-12               # BP needs finite log-likelihoods, so no prior is exactly 0


def add_heralded_loss(circuit: stim.Circuit, p_loss: float) -> stim.Circuit:
    """Copy of `circuit`, flattened, with heralded erasure after every DEPOLARIZE2.

    The kit puts DEPOLARIZE2(p) on both qubits after every CX, so this adds loss to every
    qubit of every two-qubit gate.
    """
    out = stim.Circuit()
    new_index = []   # new_index[i] = position in the new circuit of the original circuit's measurement i
    n_new = 0        # measurements emitted so far in the new circuit
    heralds = 0
    for inst in circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            # rec[-k] means "k measurements ago". The heralds we insert are extra measurements,
            # so every lookback is translated through the original measurement's absolute index.
            targets = [
                stim.target_rec(new_index[len(new_index) + t.value] - n_new) if t.is_measurement_record_target else t
                for t in inst.targets_copy()
            ]
            out.append(inst.name, targets, inst.gate_args_copy())
            continue

        out.append(inst)
        produced = _measurement_count(inst)
        new_index.extend(range(n_new, n_new + produced))
        n_new += produced

        if inst.name == "DEPOLARIZE2" and p_loss > 0:
            qubits = [t.value for t in inst.targets_copy()]
            out.append("HERALDED_ERASE", qubits, p_loss)
            n_new += len(qubits)
            for k in range(len(qubits)):
                out.append("DETECTOR", [stim.target_rec(k - len(qubits))], [HERALD, heralds])
                heralds += 1
    return out


def _measurement_count(inst: stim.CircuitInstruction) -> int:
    if not stim.gate_data(inst.name).produces_measurements:
        return 0
    single = stim.Circuit()
    single.append(inst)
    return single.num_measurements


def detector_ids(circuit: stim.Circuit, kind: str) -> np.ndarray:
    """Ids of the X-check ("X"), Z-check ("Z") or herald ("herald") detectors, in order."""
    first = HERALD if kind == "herald" else CHECK[kind]
    coords = circuit.get_detector_coordinates()
    return np.array([d for d in range(circuit.num_detectors) if coords[d][0] == first], dtype=np.int64)


class LossProblem(NamedTuple):
    """One memory basis of a lossy circuit, as a decoding problem.

    Columns of H and L are distinct error effects (which basis-check detectors and which
    observables flip), exactly like the kit's dem_matrices.
    """
    H: sp.csc_matrix              # basis-check detectors x columns
    L: sp.csr_matrix              # observables x columns
    det_ids: np.ndarray           # the circuit detectors that are rows of H
    herald_ids: np.ndarray        # the circuit detectors that are heralds
    pauli_priors: np.ndarray      # per column: probability from Pauli noise alone
    herald_given: sp.csr_matrix   # heralds x columns: P(column's effect | that herald fired)
    herald_marginal: sp.csr_matrix  # heralds x columns: P(that erasure happened and caused the effect)

    def unaware_priors(self) -> np.ndarray:
        """Priors for a decoder that knows the loss rate but not where losses happened."""
        return _combine(self.pauli_priors, self.herald_marginal)

    def aware_priors(self, fired: np.ndarray) -> np.ndarray:
        """Priors for one shot, given the indices of the heralds that fired.

        A herald that did not fire means no erasure there, so only fired heralds add anything.
        For a fired herald the erased qubit got I, X, Y or Z with equal probability; in one
        memory basis that leaves its error effect with probability 1/2.
        """
        return _combine(self.pauli_priors, self.herald_given[fired])


def _combine(base: np.ndarray, extra: sp.csr_matrix) -> np.ndarray:
    """Probability that an odd number of independent mechanisms fire, per column.

    1 - 2 P_total = (1 - 2 p_base) * product over extra mechanisms of (1 - 2 q).
    """
    log_factors = extra.tocsr(copy=True)
    log_factors.data = np.log(np.maximum(1 - 2 * log_factors.data, 1e-300))
    total = (1 - (1 - 2 * base) * np.exp(np.asarray(log_factors.sum(axis=0)).ravel())) / 2
    return np.clip(total, MIN_PRIOR, 0.5)


def loss_problem(circuit: stim.Circuit, basis: str) -> LossProblem:
    """Build H, L and the prior ingredients for decoding `basis` memory of a lossy circuit."""
    det_ids, herald_ids = detector_ids(circuit, basis), detector_ids(circuit, "herald")
    det_row = {int(d): r for r, d in enumerate(det_ids)}
    herald_row = {int(d): r for r, d in enumerate(herald_ids)}

    # Erasure is a disjoint choice (exactly one of I, X, Y, Z), which Stim only accepts as an
    # approximation by independent mechanisms. The Pauli mechanisms are unaffected.
    dem = circuit.detector_error_model(approximate_disjoint_errors=True).flattened()

    column = {}            # effect -> column index
    pauli = {}             # column -> probability, independent mechanisms merged
    herald_total = np.zeros(len(herald_ids))
    herald_effect = {}     # (herald row, column) -> summed branch probability
    for inst in dem:
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, heralds, obs = [], [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                if t.val in det_row:
                    dets.append(det_row[t.val])
                elif t.val in herald_row:
                    heralds.append(herald_row[t.val])
            elif t.is_logical_observable_id():
                obs.append(t.val)
        if len(heralds) > 1:
            raise ValueError("an error mechanism fires two heralds; independent erasures should never do that")
        if heralds:
            herald_total[heralds[0]] += p   # all branches, including those with no visible effect

        effect = (tuple(sorted(dets)), tuple(sorted(obs)))
        if effect == ((), ()):
            continue  # flips nothing the decoder sees and no observable
        col = column.setdefault(effect, len(column))
        if heralds:
            key = (heralds[0], col)
            herald_effect[key] = herald_effect.get(key, 0.0) + p   # branches are disjoint: add
        else:
            q = pauli.get(col, 0.0)
            pauli[col] = p * (1 - q) + q * (1 - p)                 # independent: exactly one fires

    n_cols = len(column)
    effects = list(column)  # dicts keep insertion order, so effects[c] is column c
    H = _incidence([e[0] for e in effects], len(det_ids))
    L = _incidence([e[1] for e in effects], circuit.num_observables).tocsr()

    pauli_priors = np.zeros(n_cols)
    pauli_priors[list(pauli)] = list(pauli.values())

    rows = np.array([h for h, _ in herald_effect], dtype=np.int64)
    cols = np.array([c for _, c in herald_effect], dtype=np.int64)
    marginal = np.array(list(herald_effect.values()))
    shape = (len(herald_ids), n_cols)
    herald_marginal = sp.csr_matrix((marginal, (rows, cols)), shape=shape)
    herald_given = sp.csr_matrix((marginal / herald_total[rows], (rows, cols)), shape=shape)

    return LossProblem(H, L, det_ids, herald_ids, pauli_priors, herald_given, herald_marginal)


def _incidence(id_lists, n_rows) -> sp.csc_matrix:
    rows = [i for ids in id_lists for i in ids]
    cols = [c for c, ids in enumerate(id_lists) for _ in ids]
    return sp.csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n_rows, len(id_lists)))
