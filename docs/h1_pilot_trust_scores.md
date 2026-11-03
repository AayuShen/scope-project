# H1 pilot: do untrained trust scores predict the world model's real error?

**Question (H1, small scale).** When DINO-WM imagines the result of a candidate plan, can a score
computed from the imagination alone tell which imagined steps are most wrong?

**Status.** Pilot: one seed, 40 decision points, released Push-T checkpoint, no training of any
kind. It is the first SCoPe-specific measurement and a check of the logging pipeline, not the
final H1 result (which needs more decision points, several seeds, the four sampling contexts of
the design doc, and the ensemble score).

Code: `scripts/h1_pilot.py` (run with `scripts/run_h1_pilot.sh`). Outputs:
`outputs/runs/h1_pilot/<tag>/summary.json` and `per_candidate.npz` (not in git); the summary used
by the deck is copied to `docs/mid_review/data/h1_pilot_summary.json`.

## Method

1. **Decision points.** 40 Push-T start/goal pairs drawn from the validation split exactly like
   DINO-WM's planner (`goal_source: dset`): a random validation trajectory, a random offset, the goal
   is the real state 25 simulator steps later. Seed 1 (the Phase 0 runs used seeds 99–101).
2. **Candidates.** CEM as in DINO-WM (300 samples, top 30, horizon 5, here 10 iterations). We record
   192 of the plans CEM evaluates: 48 random ones from iteration 1 (broad), 48 from iteration 5,
   96 from iteration 10 (concentrated near CEM's choice). So the set spans bad to good plans.
3. **Imagined vs real.** Each recorded plan is rolled out in the world model (imagined latents at
   steps 1–5) and executed in the simulator from the same start state (reset-and-execute). The real
   frames at simulator steps 5, 10, …, 25 are encoded with the same frozen encoder.
4. **Label.** Real latent error of imagined step k = mean squared difference between the imagined
   and the real patch-token latent. "Unreliable" = error in the top 10% *for that step k*, so the
   trivial fact that later steps are worse cannot inflate a score.
5. **Trust scores** (higher = less trusted; none needs training):
   - **MC-dropout**: variance of the imagined latent over 4 passes with the predictor's dropout (0.1,
     as trained) switched on.
   - **Action perturbation**: divergence of the imagined rollout when the plan's actions get small
     Gaussian noise (σ = 0.1 in normalised units).
   - **Image perturbation** (ACPC-style): divergence when the start image gets small Gaussian noise
     (σ = 0.1 in normalised pixel units).
   - **Distance to training data**: mean distance from the imagined latent to its 5 nearest encoded
     training frames (300 training trajectories × 15 frames, random projection to 512 dimensions).
   - **Imagined cost**: the plan's own predicted distance to the goal.
   - Controls: **distance moved** (how far the imagined latent is from the start; trivial),
     **random**.
6. **Metrics.** AUROC for the "unreliable" label, averaged over steps 1–5; Spearman correlation with
   the raw error; 95% bootstrap intervals over decision points (500 resamples). Also: Spearman
   agreement between the imagined and the real cost ranking of the candidates, per decision point.
7. **Offline gate check.** Per decision point, among its 192 recorded candidates: plain selection
   picks the lowest imagined cost; a gate first rejects the 30% with the highest score (worst step
   of the plan), then picks the lowest imagined cost. Compared with a random gate (rejects 30% at
   random) and an oracle gate (rejects using the true error). Outcome: does the chosen plan succeed
   in the simulator? Offline and on recorded candidates only, so it is a hint for H4, not H4.

## Results

Run `pilot40`, 3 November 2026: 40 decision points × 192 candidates × 5 imagined steps =
**38,400 imagined steps** checked against the simulator, about 2 minutes per decision point on the
RTX 5060 (81 minutes in total). Exit code 0, no skipped decision points.

### Does a score flag the most wrong imagined steps? (H1)

| Score | AUROC (top-10% error, per step) | 95% interval | Spearman ρ with error |
|---|---|---|---|
| **MC-dropout** | **0.869** | 0.843–0.894 | 0.66 |
| Action perturbation | 0.656 | 0.631–0.678 | 0.27 |
| Distance to training data | 0.646 | 0.588–0.700 | 0.14 |
| Imagined cost | 0.505 | 0.467–0.545 | 0.02 |
| Image perturbation | 0.498 | 0.432–0.563 | −0.17 |
| *Distance moved (trivial control)* | 0.796 | 0.757–0.831 | 0.50 |
| *Random (control)* | 0.498 | 0.489–0.505 | −0.01 |

- **MC-dropout passes the go / no-go bar of the design** in this pilot: its 95% interval lies
  entirely above both random and the trivial "distance moved" control (0.84 > 0.83). It is the
  only score that does.
- The trivial control is strong (0.80): within one imagined step, plans that move the scene a lot
  are the ones the model gets wrong. Any trust score has to be compared with it, not only with
  random, which is why it is on the slide.
- Action perturbation and distance to training data carry some signal but less than the trivial
  control. The plan's own imagined cost carries none, so "the planner looks confident" is not
  evidence that the imagination is right.
- Image perturbation (σ = 0.1 on the start image) is at chance and slightly anti-correlated with the
  raw error. The frozen DINOv2 encoder absorbs this noise, so the perturbation mostly measures the
  encoder, not the predictor. A larger or structured perturbation is a follow-up, not a conclusion.

### Context numbers

- Real latent error grows with the horizon: median 0.056, 0.071, 0.084, 0.099, 0.116 at imagined
  steps 1–5 (hence the per-step label).
- Imagined and real cost rank the candidates of a decision point similarly (mean Spearman 0.76), so
  the model is mostly right about which plans are better. The failures are in the tail.
- 6.3% of all recorded candidates solve the task when executed open-loop: 0.3% of the iteration-1
  candidates, 4.6% of iteration 5 and 10.2% of iteration 10, so the set does span bad to good
  plans. Plain selection picks an iteration-10 candidate in 90% of decision points. CEM's mean plan
  after 10 iterations solves 70% of decision points (Phase 0 with 30 iterations: 87%).

### Offline gate check (a hint for H4, not H4)

| Selection among the 192 recorded candidates | Chosen plan succeeds |
|---|---|
| Plain (lowest imagined cost) | 65.0% |
| Oracle gate (true error) | 65.0% |
| Random gate | 60.0% |
| MC-dropout gate | 65.0% |
| Action-perturbation gate | 72.5% |
| Image-perturbation gate | 65.0% |
| Distance-to-data gate | 62.5% |
| Distance-moved gate | 60.0% |

With 40 decision points one decision point is 2.5 points, so none of these differences is
meaningful. The informative row is the oracle: even rejecting by the *true* error does not change
the chosen plan's success here. Among the recorded candidates, the lowest-imagined-cost plan is
rarely one of the high-error ones, so a one-shot gate has little to change. Whether gating helps
must be measured in the closed loop over several replanning rounds and in harder conditions (H4),
which is the next step; this pilot does not predict its outcome.

### What this pilot does and does not show

- Shows: on Push-T, an untrained MC-dropout score ranks the most wrong imagined steps well above
  chance and above the trivial control, with the pipeline (reset-and-execute, per-step labels,
  bootstrap over decision points) working end to end.
- Does not show: that the result holds over seeds, sampling contexts or environments; that the
  score is calibrated (ECE not computed yet); or that gating improves real success.
- Next: 3 seeds and ≥ 100 decision points, the remaining sampling contexts of the design doc, an
  ensemble score, then the closed-loop gate against plain, random and oracle gates.
