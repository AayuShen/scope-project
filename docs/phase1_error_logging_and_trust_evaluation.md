# Phase 1–2 design: logging model error, and testing trust scores against it

Status: draft for review, no code yet. Written while Phase 0 runs; builds on
`src/scope/world_model/interface.py` and `src/scope/trust/interface.py`.

## Principle

A trust score is a hypothesis about where the world model is wrong. It is never treated as
physical truth: an unfamiliar latent can be predicted correctly, and a familiar one can be
predicted badly. Everything below exists to measure that, against what the simulator
actually does.

## 1. What "wrong" means (two separate labels)

The simulator lets us reset to any recorded state and execute any action sequence, so for a
candidate plan from real state `s_t` we can compare the model's imagined future with the
real one.

| Label | Definition | Why separate |
|---|---|---|
| **Latent error** `e_k` | `latent_distance(imagined z_{t+k}, encode(real obs_{t+k}))` for k = 1..H | Measures prediction quality, whether or not it matters for the task |
| **Task consequence** | Does the plan, executed for real, get closer to the goal? Concretely: real goal distance after H steps, and whether the model's predicted cost ranking of candidates matches the real ranking | A prediction can be inaccurate yet still rank plans correctly, or accurate yet unhelpful |

Planning failure in an episode (the headline label) is "never solved within the round budget".
Error and consequence are recorded per candidate, per step.

## 2. What Phase 1 records (per CEM call, per episode)

For one decision point (real state `s_t`, goal `g`):
- the real observation, its encoding, the goal encoding;
- the candidate action sequences actually evaluated (CEM's samples, and the elite set);
- for each recorded candidate: imagined latents for k = 1..H, the model's cost;
- for a subsample of candidates (all elites, plus a random draw of the rest): the real
  rollout from `s_t` (reset-and-execute), its encoded latents, and its real goal distance.

Storage: latents are 256 × 384 floats per frame (≈0.4 MB in fp32; fp16 halves it). We store
fp16 and a small subsample, which is what F: is for; summaries (per-step errors, scores,
costs) are tiny and stay on E:. Format: one `.npz` shard per episode plus a JSON index.

Contexts to sample, so that errors are not collected only where the planner is comfortable:
1. on-policy: states visited by the MPC planner;
2. dataset states with random actions (off-distribution actions);
3. perturbed states (shifted block position, never-seen configurations);
4. visual shift that leaves dynamics unchanged (e.g. recoloured block), the
   "unfamiliar but correct" case.

## 3. Candidate trust scores (Phase 2, offline, no planner involved yet)

Each implements `TrustScorer.uncertainty(rollout) -> (B, H)`; higher = less trusted.

| Score | Idea | Needs |
|---|---|---|
| Ensemble disagreement | Variance across several predictors | Training a small ensemble (frozen encoder, so cheap on cached features) |
| MC-dropout | Variance under dropout. Note: released Push-T predictor loads with dropout off, so it must be switched on deliberately | Nothing new |
| Latent density / kNN distance | Distance of the imagined latent to the nearest training latents | Index over cached training latents (the large store on F:) |
| Self-consistency | Predict the same step from different history windows or action splits | Nothing new |
| Predicted-cost margin | Gap between the best and later candidates | Nothing new |

## 4. Hypotheses and the measurements that test them

| | Question | Measurement | Controls |
|---|---|---|---|
| H1 | Does the score predict real latent error? | AUROC / AUPRC for "error above the 90th percentile"; Spearman correlation; reliability curve of binned score against mean error; expected calibration error | A random score; the distance to the start latent (a trivial baseline) |
| H2 | How often does it reject plans that would have worked? | False rejection rate: rejected candidates whose real rollout still reaches the goal | Random gate at the same rejection rate |
| H3 | How often does it accept plans that fail? | False acceptance rate | Same |
| H4 | Does gating improve real planning? | Success per MPC round, failure rate, compute per episode | Plain MPC; soft penalty; random gate; **oracle gate** (uses the true error) |

The oracle gate separates "the score is poor" from "gating cannot help even with a perfect
score". If the oracle gate does not beat plain MPC, the idea is limited by something other
than the quality of the score, and we find that out before building the full loop.

Statistics: several seeds per setting, report mean and spread, and bootstrap confidence
intervals over episodes. Threshold choice must not use the test episodes (calibrate the
rejection threshold on held-out decision points).

## 5. Recovery loop (Phase 3, for reference)

Observe the real state → sample candidates → score every imagined step → **reject** a
candidate if any step in its horizon fails the threshold → choose from accepted candidates
only → execute the first accepted action(s) → observe again. Imagined latents are never used
as the starting state of the next plan.

If too few candidates are accepted: widen the search, then shorten the horizon (H → H/2 → …),
then take a safe action (no-op); log which step of this ladder fired.

## 6. Decisions needed before coding

1. **Real-rollout budget.** Reset-and-execute is cheap per candidate (milliseconds) but the
   count scales with episodes × decision points × candidates; the subsample sizes above are a
   starting point.
2. **Which contexts matter most** for the paper's story (§2 list), given limited compute.
3. **Ensemble members** are cheap only if trained on cached encoder features; this needs the
   cache of frozen-encoder features, which is the first thing to build.
