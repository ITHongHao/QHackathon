"""Tests for the heralded-loss model and the loss-aware decoding inputs.

  python -m pytest test_loss.py
"""

import pathlib
import sys

import numpy as np
import pytest

KIT = pathlib.Path(__file__).resolve().parent.parent / "MidQWest-2026" / "challenge-1-ml-decoder-kit"
sys.path.insert(0, str(KIT))

from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit  # noqa: E402
from make_circuit import memory_experiment  # noqa: E402

from loss import HERALD, add_heralded_loss, detector_ids, loss_problem  # noqa: E402


def short_circuit(basis, p=0.004):
    """Gross code memory experiment with 2 noisy cycles: same structure, much faster to analyse."""
    return memory_experiment(code="gross", basis=basis, rounds=2, p=p)


def pauli_mechanisms(dem, keep_detector):
    """{(detectors, observables): probability} of the error mechanisms that touch no herald.

    keep_detector maps a detector id to its id in the original circuit, or None for a herald.
    """
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [keep_detector(t.val) for t in inst.targets_copy() if t.is_relative_detector_id()]
        if None in dets:
            continue
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        out[(tuple(sorted(dets)), obs)] = inst.args_copy()[0]
    return out


@pytest.mark.parametrize("basis", "XZ")
def test_lookbacks_survive_inserted_heralds(basis):
    """Dropping the erasure mechanisms must give back exactly the original error model.

    Every original detector compares earlier measurements through rec[-k]. If the
    remapping around the inserted herald measurements were off by even one, detectors
    would point at the wrong measurements and the Pauli mechanisms would change.
    """
    original = short_circuit(basis)
    lossy = add_heralded_loss(original, 0.01)
    non_herald = {int(d): i for i, d in enumerate(np.setdiff1d(np.arange(lossy.num_detectors),
                                                               detector_ids(lossy, "herald")))}
    assert len(non_herald) == original.num_detectors

    expected = pauli_mechanisms(original.detector_error_model(), lambda d: d)
    got = pauli_mechanisms(lossy.detector_error_model(approximate_disjoint_errors=True), non_herald.get)
    assert got.keys() == expected.keys()
    assert np.allclose([got[k] for k in expected], list(expected.values()), rtol=1e-12)


def test_noiseless_lossy_circuit_is_silent():
    lossy = add_heralded_loss(short_circuit("X", p=0.0), 0.0)
    det, obs = lossy.compile_detector_sampler(seed=0).sample(50, separate_observables=True)
    assert not det.any() and not obs.any()


def test_gross_circuit_gets_one_herald_per_gate_qubit():
    lossy = add_heralded_loss(load_circuit(GROSS_CIRCUITS["X"], 0.004), 0.001)
    heralds = detector_ids(lossy, "herald")
    assert len(heralds) == 20736                     # 864 CX per cycle x 2 qubits x 12 cycles
    assert len(detector_ids(lossy, "X")) == 936      # check detectors unchanged
    assert len(detector_ids(lossy, "Z")) == 792
    coords = lossy.get_detector_coordinates()
    assert [coords[int(d)] for d in heralds[:3]] == [[HERALD, 0.0], [HERALD, 1.0], [HERALD, 2.0]]


@pytest.mark.parametrize("basis", "XZ")
def test_without_loss_the_problem_equals_the_kits(basis):
    circuit = short_circuit(basis)
    ours = loss_problem(add_heralded_loss(circuit, 0.0), basis)
    H, L, priors = dem_matrices(circuit, basis)

    def by_effect(H, L, priors):
        H, L = H.tocsc(), L.tocsc()
        return {(tuple(H[:, c].indices), tuple(L[:, c].indices)): priors[c] for c in range(H.shape[1])}

    a, b = by_effect(ours.H, ours.L, ours.unaware_priors()), by_effect(H, L, priors)
    assert a.keys() == b.keys()
    assert np.allclose([a[k] for k in b], list(b.values()), rtol=1e-9)


@pytest.mark.parametrize("basis", "XZ")
def test_fired_herald_makes_its_effect_a_coin_flip(basis):
    """Given a fired herald, the erased qubit got I, X, Y or Z with probability 1/4 each.
    In one memory basis two of those are visible, so each herald's effect has probability 1/2."""
    problem = loss_problem(add_heralded_loss(short_circuit(basis), 0.002), basis)
    given = problem.herald_given
    assert np.allclose(given.data, 0.5)
    assert np.all(np.diff(given.indptr) <= 1)        # at most one visible effect per herald
    # A shot where herald 0 fired: its column becomes a coin flip, everything else is untouched.
    fired = np.flatnonzero(np.diff(given.indptr))[:1]
    col = given[fired].indices[0]
    priors = problem.aware_priors(fired)
    assert priors[col] == pytest.approx(0.5)
    others = np.setdiff1d(np.arange(len(priors)), [col])
    assert np.allclose(priors[others], np.clip(problem.pauli_priors[others], 1e-12, 0.5))
