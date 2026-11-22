# Novel contributions

MidQWest 2026, Challenge 1: an ML-assisted decoder for the Gross code [[144,12,12]].

This file names the two things we claim as new and says how we will test each one. Both
make use of the **loss flags (heralds)**: they come almost free on neutral-atom hardware,
and our baselines show they are worth an order of magnitude in error rate.

| # | Contribution | What it improves | One-line claim |
|---|---|---|---|
| 1 | Loss-aware router | **Speed** (mean and p99 latency) | A cheap learned router that uses where and when losses happened decides which shots need relay-BP, and relay-BP no longer pays a per-shot rebuild cost for loss. |
| 2 | Learned loss-aware priors for relay-BP | **Accuracy** (logical error rate) | A small network reads the heralds and the syndrome and gives relay-BP better per-shot error priors than the fixed "erased location gets 1/2" rule. |

"Novel" here means "not found in our literature scan on 2026-11-22" (see References), not
proven absence. The closest prior work is listed for each idea.

---

## Shared setup

### Where we start (the `vibez` branch)

- **Pipeline.** For each shot and memory basis: loss-aware BP (80 iterations). If it
  converges, accept. Otherwise OSD-0 gives an answer, and a logistic-regression router
  (9 features, 10 parameters) decides whether to trust it or send the shot to loss-aware
  relay-BP (`two_stage.py`, `router.py`).
- **Loss model.** Heralded erasure with immediate replacement after every two-qubit gate,
  `p_loss = p/2` (`loss.py`). A loss-aware decoder sets the prior of each erased location's
  error mechanisms to about 1/2.
- **Router result** on 2,000 test pairs per p (`results/evaluation_loss0.5.csv`):

| p | relay-BP alone: rate per cycle, mean / p99 ms | router (logistic): rate per cycle, mean / p99 ms | escalated | no-ML control "BP > relay-BP": mean ms |
|---|---|---|---|---|
| 0.003 | 8.3e-5, 88 / 698 | 1.3e-4, 42 / 459 | 2.8% | 66 |
| 0.004 | 1.81e-3, 364 / 4,020 | 1.89e-3, 294 / 4,026 | 18.7% | 350 |
| 0.005 | 1.48e-2, 1,455 / 6,684 | 1.50e-2, 1,399 / 6,730 | 52.2% | 1,461 |

The router matches relay-BP's accuracy (paired test p-values 0.42 to 1.0) and cuts the
mean latency, but **the slowest 1% of shots don't improve**, and at p = 0.004 and 0.005
the saving is small. The cause is relay-BP's own heavy tail on the escalated shots, plus a
fixed per-shot cost: relay-BP has no "update priors" call, so `timed_relay` rebuilds it
for every shot with loss (about 20 to 33 ms per shot).

### Baselines both contributions must beat

The fair baseline for a decoder that sees the heralds is a decoder that also sees them.
Beating flag-blind decoders would only show that the flags help, not that ML does.

1. **Loss-aware relay-BP** (`loss_baselines.py`, `relay-aware`). In the 100-pair smoke
   test at p = 0.004 it was about 18x better than flag-blind relay-BP (2 vs 35 failures).
   This is the main bar.
2. **Loss-aware BP-OSD** at the kit's settings (`bposd-aware`), about 7x better than
   flag-blind.
3. **MMM-style loss-aware BP** (Murri, arXiv:2610.38283). Per shot, it builds the loss
   hypergraph as a mixture over the likely correlated-loss scenarios (for example, *when*
   between two heralds an atom was lost) and runs BP on it. Reported: up to 25x better
   than loss-blind, 2x better than a uniform loss mixture. We implement a simple version:
   for each fired herald, spread the erasure over the gate slots since the qubit's last
   good herald, weighted by the loss rate. **Note:** under our current
   immediate-replacement model this equals baseline 1, because loss has no duration. It
   only differs once loss lasts several gates, which is one more reason to move to clifft
   (below).
4. **No-ML controls** (already in `evaluate_router.py`): "BP > relay-BP", "BP > OSD-0",
   and a disagreement-only router (escalate when BP's guess and OSD-0 disagree;
   Wang and Coveney, arXiv:2609.37629).

### Move to clifft LOSS with heralds

The current loss model is optimistic: a lost atom is replaced at once, and loss happens
only at gates. Before the final numbers, data generation moves to clifft's `LOSS`
instruction with heralds (Unitary Foundation; loss is an experimental API since v0.7.0).
There, a lost atom stays missing until detected, and the checks it touches become
unknown rather than 0 or 1. We keep the Stim model as a fast development path, and report
both, labelled. Decoder interfaces don't change: each shot gives a syndrome plus a list of
fired heralds (with their round and qubit).

### Measurement protocol (both contributions)

- **Sweep** p = 0.003, 0.004, 0.005; loss ratio `p_loss/p` = 0.5 by default, plus 0.25
  and 1.0 as a sensitivity check.
- **Shots.** At least 2,000 test pairs per point; 20,000 at p = 0.003, where loss-aware
  relay-BP fails only about 2 times in 2,000. Train, validation and test use disjoint
  seeds (`SEED_BASE` in `make_router_data.py`).
- **Metrics.** Logical error rate per cycle (kit convention) with standard errors;
  latency per pair, mean, median, p99 and max, on one performance core; resources
  (parameters, peak memory per worker, training time, and inference cost per shot).
- **Significance.** Same shots for every decoder, paired exact test on the pairs where
  only one of two decoders fails (the `paired_pvalue_vs_relay` column already computed).
- **Plots.** Error rate vs p, mean and p99 latency vs p, and the router's accuracy vs
  latency tradeoff curve (threshold sweep).

---

## Contribution 1: Loss-aware router (speed)

### Motivation

Published routers and confidence-based decoders on the Gross code use no loss
information. CoDecoder (arXiv:2610.28748) routes unsure shots from a neural frontend to
relay-BP; graph neural post-selection (arXiv:2610.00504) predicts failures from syndromes
and BP posteriors; Wang and Coveney (arXiv:2609.37629) escalate on BP / OSD-0
disagreement. All are loss-free. Under loss, *where* the losses fell is strong
information about whether a fast answer can be trusted: a loss next to an unsatisfied
check, or on a qubit that the OSD-0 answer relies on, is a warning sign.

Our current router already sees two coarse loss features: the number of heralds and the
number of OSD-0 mechanisms sitting on erased locations. The contribution is to use the
**pattern** of losses, to **fix the per-shot relay-BP rebuild**, and to **target the p99
tail**, which the current router leaves untouched.

### Method

1. **Richer loss features** (all cheap, computed from data we already have per shot):
   - heralds whose error mechanisms touch a check that BP left unsatisfied;
   - heralds in the last two rounds (late losses have fewer later checks to reveal them);
   - largest number of heralds touching any one check, and number of checks touched by
     two or more heralds (clustered loss);
   - prior-weighted overlap between the erased mechanisms and the OSD-0 answer, split by
     whether they flip a logical observable (from `L`);
   - herald count in X vs Z sectors relative to the memory basis.
2. **Remove the relay-BP rebuild.** Options, in order of preference:
   (a) add a prior-update method to the relay-bp Python binding (it is built from source
   with Rust already), so one decoder object is reused like BP-OSD's
   `update_channel_probs`; (b) if that is not possible in time, build relay-BP once with
   the flag-blind priors and pass the loss information as a per-shot prior override if
   the API offers one; (c) as a last resort, keep the rebuild but count it separately and
   report it. Check first that (a) gives bit-identical outputs to the rebuilt decoder on
   the same shots.
3. **Three-way routing for the tail.** Instead of {accept OSD-0, run relay-BP}, the
   router picks among {accept OSD-0, run BP-OSD order 10, run relay-BP}. Loss-aware
   BP-OSD has a bounded tail (p99 about 660 ms at p = 0.004) and fixes most OSD-0
   mistakes; relay-BP is kept for the shots the router thinks only it can save. As a
   second lever, give relay-BP a time or leg budget on escalated shots and fall back to
   the BP-OSD answer when it runs out.
4. **Models.** Keep logistic regression as the headline (a decision costs microseconds,
   every weight can be explained). Fit the small gradient-boosted model as an upper
   bound on what the features can give.

### Inputs and outputs

- **Inputs per shot and basis:** the syndrome, the fired heralds (index, round, qubit),
  and the BP / OSD-0 output (hard decision, posterior LLRs, convergence flag). Features
  are the 9 existing ones plus the loss-pattern ones above (about 15 to 20 numbers).
- **Output:** probability that the OSD-0 answer is wrong (two-way), or a choice of
  stage (three-way, via two logistic models in a row: "OSD-0 wrong?", then "BP-OSD
  wrong?"). A threshold per p is set on validation data by the accuracy budget
  (`choose_threshold`).

### Training data

`make_router_data.py --split train` already gives 6,000 pairs per p with features and
"OSD-0 wrong" labels from simulator ground truth (the true observable flips). Add the
new features there, and for the three-way router also run loss-aware BP-OSD on the train
split to label "BP-OSD wrong". Regenerate with clifft for the final numbers. Training
takes seconds.

### Evaluation plan

- Compare on the same test shots: relay-BP alone (rebuilt and reused), BP-OSD alone, the
  no-ML controls, the current 9-feature router, the loss-aware router (two-way and
  three-way), and the GBM upper bound.
- **Ablations:** (i) rebuild fix only, no router change; (ii) new loss features only;
  (iii) three-way routing only. This shows which part gives which saving.
- Report a **feature importance** table (standardised logistic weights) to show the loss
  features carry weight.
- Plot the accuracy vs mean latency and accuracy vs p99 curves across thresholds.

### Success criteria

- Error rate within one standard error of loss-aware relay-BP at every p, and paired
  test p-value above 0.05 (no significant loss of accuracy).
- **Mean latency** below the current router at every p, and at least 30% below
  loss-aware relay-BP alone at p = 0.004 and 0.005.
- **p99 latency** at least 2x lower than relay-BP alone at p = 0.004 and 0.005 (today it
  is unchanged).
- The ML router beats the best no-ML control at equal accuracy. If it doesn't, we say so:
  "routing helps; the ML part doesn't add beyond simple rules" is still a finding.

### Risks

- **Herald count may already explain most of it**, so the new features add little. The
  ablation will show this; the rebuild fix and three-way routing still give speed.
- **The rebuild fix may need Rust changes** in relay-bp. Fallback (c) above keeps the
  result honest.
- **Shifting the tail to BP-OSD can cost accuracy** at p = 0.005, where BP-OSD is 2.6x
  worse than relay-BP. The threshold budget controls this.
- **Timing noise.** Measure on idle performance cores only, as noted in `BASELINES.md`.
- **Model drift with clifft.** Features tied to immediate replacement (for example "late
  losses") may behave differently with delayed loss; retrain on clifft data.

---

## Contribution 2: Learned loss-aware priors for relay-BP (accuracy)

### Motivation

Priors are the lever on this code. PriorTilt (arXiv:2611.13373) beats relay-BP by 4.3x on
the Gross code at p = 0.001 by perturbing under 0.5% of BP priors, by hand. MMM shows
that a better per-shot loss model gives another 2x over uniform loss priors. Our own
herald rule (erased location gets 1/2) is a fixed formula: it ignores that a loss next to
other losses, or next to a burst of detection events, changes which mechanisms likely
fired. Neural priors exist without loss (for example Astra GNN + OSD), and loss priors
exist without learning (MMM, our herald rule). **No paper we found trains a model on loss
flags for a bivariate bicycle code under circuit-level noise.** The closest ML-with-loss
work is a spatiotemporal GNN for surface codes that infers loss from syndromes without
heralds (arXiv:2604.14269), and PRISM (arXiv:2611.13630), a transformer with erasure
input for photonic fusion protocols.

### Method

A small model outputs a **per-shot prior for every error mechanism** (every column of
`H`), then loss-aware relay-BP runs with those priors unchanged otherwise.

1. **Version A, per-mechanism MLP (start here).** For each column, a feature vector:
   its Pauli prior and herald-rule prior (`aware_priors`), the number of fired heralds
   that can cause it and within one or two hops on the Tanner graph, the number of its
   detectors that fired, the detection-event count in its neighbourhood, its round, and
   whether it flips a logical. A shared 2-layer MLP (a few thousand weights) outputs a
   correction to the log-odds of the herald-rule prior. Same weights for every column,
   so the model is small and respects the code's symmetry.
2. **Version B, GNN on the Tanner graph (stretch).** Message passing between mechanism
   nodes and detector nodes for 3 to 5 rounds, with heralds as node inputs. It is the
   natural way to let nearby losses and detection events inform each other.
3. **Avoid double counting.** The model sees the syndrome, and relay-BP will see it again.
   Output priors are therefore mixed with the herald-rule priors with a weight α, and
   optionally scaled by a temperature, both tuned on validation error rate. α = 0
   recovers the baseline exactly, so the learned model can never be forced on.
4. **Fallback if per-mechanism learning fails:** learn only a handful of knobs (for
   example, the prior given to an erased location, a boost for mechanisms next to
   clustered losses, a temperature) by black-box search on validation error rate.

### Inputs and outputs

- **Inputs per shot and basis:** the syndrome (detection events per detector), the fired
  heralds (index, round, qubit), and the static structure (`H`, `L`, Pauli priors, the
  herald-to-mechanism map from `loss_problem`).
- **Output:** a prior probability for each column of `H` (a few thousand numbers), fed to
  relay-BP for that shot.

### Training data

- **Labels from simulator ground truth.** Sample from the detector error model with
  Stim's `compile_sampler().sample(..., return_errors=True)`, which returns which error
  mechanisms fired. Map each DEM error to its column, and the label for a column is the
  parity of its fired mechanisms. Train with binary cross-entropy against those labels.
  The DEM treats the disjoint erasure branches as independent (an approximation, already
  used in `loss.py`), so **evaluation always uses circuit-level sampling**, never DEM
  samples.
- **With clifft:** take the per-shot fault record if clifft exposes one; otherwise build
  labels from the Stim DEM of an equivalent delayed-loss circuit.
- **Size.** About 100,000 shots per p per basis (sampling is fast; no decoding is needed
  to make labels). Labels are very sparse, so weight positives or use focal loss.
- **Degeneracy.** Many error patterns give the same syndrome and logical effect, so the
  label is noisy by nature. The model learns marginal probabilities, which is what a
  prior should be.

### Evaluation plan

- Compare on the same test shots: loss-aware relay-BP (herald rule), MMM-style BP and
  MMM-style priors into relay-BP, flag-blind relay-BP, loss-aware BP-OSD, and relay-BP
  with learned priors (Version A, and B if done).
- **Ablations:** model inputs restricted to heralds only (should roughly recover the
  herald rule), heralds + syndrome, and + BP posteriors from a first BP pass; α = 0 vs
  tuned α.
- **Calibration:** reliability plot of predicted vs actual flip rate per column.
- **Cost:** model inference time per shot added to relay-BP time; also check whether
  better priors let relay-BP converge in fewer legs (this could cut latency too).
- **Combine with contribution 1:** the learned priors can feed every stage (BP, OSD-0,
  relay-BP) of the routed pipeline.

### Success criteria

- **Main:** lower logical error rate than loss-aware relay-BP at p = 0.004 and 0.005, with
  a paired p-value below 0.05 on the same shots; a target of at least 1.3x fewer failures.
- At least matches the MMM-style baseline under clifft delayed loss.
- Never worse than the herald rule at any p (guaranteed by tuning α, with α = 0 as floor).
- Model inference adds under 10% to mean decoding time.

### Risks

- **The herald rule may already be near-optimal** under immediate replacement, where a
  loss is just a known erasure. The gain is more likely under clifft delayed loss and
  correlated loss, so that is where to test first.
- **Too few failures to measure.** At p = 0.003 loss-aware relay-BP fails a few times per
  thousand pairs; use 20,000 pairs there, and lead with p = 0.004 and 0.005.
- **Double counting the syndrome** can make relay-BP overconfident; α and temperature
  tuning address it.
- **Training distribution.** A model trained at one p and loss ratio may not carry over;
  train on the mix of all three p and test per p, plus one held-out loss ratio.
- **DEM labels vs circuit noise** mismatch from the erasure approximation; evaluation on
  circuit samples keeps the reported numbers honest.

---

## Timeline (hackathon)

1. Fix the relay-BP rebuild and rerun the router test (contribution 1, half a day).
2. Add loss-pattern features and three-way routing; ablations and plots (half a day).
3. Label generation and Version A of learned priors; α tuning; evaluation at p = 0.004
   and 0.005 (one day).
4. Switch data generation to clifft LOSS; rerun both contributions and the MMM-style
   baseline (as time allows).
5. Full sweep, plots and slides.

---

## References

All are arXiv preprints unless marked peer reviewed (PR). Status read on 2026-11-22.

**Code and baselines**
- Bravyi et al., High-threshold and low-overhead fault-tolerant quantum memory, *Nature*
  627, 778 (2024), PR. https://doi.org/10.1038/s41586-024-07107-7 (the Gross code)
- Müller et al., Improved belief propagation is sufficient for real-time decoding of
  quantum memory (relay-BP). https://arxiv.org/abs/2506.01779
- Challenge kit: https://github.com/Infleqtion/MidQWest-2026
- clifft (Unitary Foundation), loss and leakage simulation: https://github.com/unitaryfoundation/clifft
- Stim: https://github.com/quantumlib/stim ; ldpc (BP-OSD): https://pypi.org/project/ldpc/ ;
  qLDPC: https://github.com/qLDPCOrg/qLDPC

**Loss-aware decoding (baselines for both contributions)**
- Murri, MMM decoder (mixture over correlated-loss scenarios on the Gross code).
  https://arxiv.org/abs/2610.38283
- Baranes et al., delayed-erasure decoding (surface code). https://arxiv.org/abs/2502.20558
- Perrin, Roger, Pupillo, loss graph decoding. https://arxiv.org/abs/2603.24237
- Bluvstein et al., heralds and ML decoding in neutral-atom hardware, *Nature* (2025), PR.
  https://doi.org/10.1038/s41586-025-09848-5

**Routing and confidence (contribution 1 prior work, all loss-free)**
- Guang et al., CoDecoder: neural frontend plus relay-BP fallback on the Gross code.
  https://arxiv.org/abs/2610.28748
- Carty, Noszko, Roffe, Bondesan, graph neural post-selection.
  https://arxiv.org/abs/2610.00504
- Wang and Coveney, disagreement routing between BP and OSD-0.
  https://arxiv.org/abs/2609.37629
- Meinking, Takou, Brown, hybrid sliding-window decoding with confidence switching.
  https://arxiv.org/abs/2611.21906
- Chamberland et al., Techniques for combining fast local decoders with global decoders
  under circuit-level noise, *Quantum Sci. Technol.* 8, 045011 (2023), PR.
  https://doi.org/10.1088/2058-9565/ace64d

**Priors and learned models (contribution 2 prior work)**
- Ye and Delfosse, PriorTilt. https://arxiv.org/abs/2611.13373
- Maan et al., Astra GNN decoder, *npj Quantum Information* (2025), PR.
  https://github.com/arshpreetmaan/astra
- Liu and Poulin, Neural belief-propagation decoders, *Phys. Rev. Lett.* 122, 200501
  (2019), PR. https://doi.org/10.1103/PhysRevLett.122.200501
- Wang et al., STGNN loss detection (surface code, no heralds).
  https://arxiv.org/abs/2604.14269
- PRISM (PsiQuantum), transformer with erasure input for fusion-based protocols.
  https://arxiv.org/abs/2611.13630
- Gu, Bonilla Ataides, Lukin, Yelin, Cascade neural decoders.
  https://arxiv.org/abs/2604.08358

**Project notes:** frontier map https://claude.ai/artifact/VEpQ1X9MQwyBssWXqeV3db and
research brief https://claude.ai/artifact/B4gftQUrqsZ3ZSjayEBn9W.
