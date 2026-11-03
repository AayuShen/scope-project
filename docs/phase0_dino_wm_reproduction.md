# Phase 0 — Reproduce DINO-WM planning

**Goal.** Get DINO-WM's released checkpoints planning at their reported success
rates on this machine before anything of ours touches the loop. Every later
comparison depends on this baseline being right.

**Pinned upstream:** `external/dino_wm` @ `0a9492f` (2025-03-24).

## Scope

| Env | Checkpoint | Dataset | Simulator | Order |
|---|---|---|---|---|
| Push-T | `pusht` | `pusht_noise` (2.8 GB) | pymunk | 1st |
| Wall | `wall` | `wall_single` (1.7 GB) | custom 2D | 2nd |
| PointMaze | `point_maze` | `point_maze` (0.7 GB) | MuJoCo 2.1 | 3rd |

Deformable envs (Rope, Granular) are out of scope: they need PyFleX via Docker
and ~16 GB of data, and add nothing Phase 1 needs.

## Protocol

1. Smoke test: `n_evals=5` per env; confirm the run completes and videos are written.
2. Full run: upstream planning configs unchanged (`plan_<env>.yaml`: MPC-CEM,
   horizon 5, 300 samples, top-30, 5 actions executed per replan), `n_evals=50`, seed 99.
3. Three seeds for the env we carry into Phase 1.
4. Record the success rate against the paper's number for the *same* planner
   setting; note the paper table and row next to each result.

**Pass:** each env within the seed-to-seed spread of the reported number.
**If not:** diagnose before Phase 1. Check the deviations below first.

## Deviations from upstream

| Change | Reason | Risk to fidelity |
|---|---|---|
| torch 2.3.0 → 2.7.1 (cu128) | RTX 5060 (sm_120) has no kernels in 2.3 | Numerical drift; low |
| Python 3.9 → 3.10 | Shared env with `scope` package | Low |
| mujoco-py compiled with system gcc 15 using `-std=gnu17` and pointer-type errors downgraded to warnings | conda gcc cannot see Ubuntu OSMesa headers; gcc 14+ rejects its 2021-era generated C | None (compile flags only) |
| Lean dependency set (no TF / JAX / robosuite) | Unused on the planning path | None if imports resolve |
| `WANDB_MODE=disabled` (+ git lookup skipped) | No W&B project yet; wandb's git probe fails on /mnt/e | None |
| Runtime patches via `scripts/plan_dino_wm.py` (`src/scope/world_model/dino_wm_patches.py`): fused attention, encoder dedup, rollouts in chunks of 150, **TF32 matmuls** | Upstream needs >8 GB and ~26 s per 300-candidate rollout here | Latent drift ≤0.7% (noise floor 0.1–0.2%); **identical CEM elites and updates** in every benchmark case. See *Speed-ups* below |
| PyTorch capped at 85% of VRAM | Windows spills overflow into system RAM silently (runs 2–3× slower) | None; overflow now raises OOM |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | torch ≥2.6 refuses checkpoints that pickle whole modules | None (restores torch 2.3 loading; trusts the sha256-verified official checkpoints) |
| Smoke test only: `planner.max_iter=2` | Upstream `max_iter: null` replans with no cap | Smoke test only; full runs use upstream settings under `timeout` |

## Hardware notes

- 8 GB VRAM. CEM loops over eval instances serially with 300 samples each, so
  peak memory is one instance's batch, not all 50 at once. Smoke test confirms this.
- Push-T and PointMaze spawn one subprocess env per eval (`SubprocVectorEnv`).
  At `n_evals=50` this is RAM-heavy on a laptop; lower `n_evals` and run
  several seeds if needed.

## Observations for later phases (from reading the planner)

- Upstream MPC executes **all 5** planned actions before replanning, and
  replays the full action history from the initial state to get feedback. Our
  recovery loop executes only accepted actions and re-observes, so Phase 3 needs
  its own step-wise control loop rather than wrapping `MPCPlanner`.
- `CEMPlanner` has no notion of rejection; it returns every plan as valid.
- `VWorldModel.rollout` re-encodes the observation for each of the 300
  samples. The adapter should encode once and share the context.
- The released Push-T predictor loads in eval mode (no active dropout), so
  rollouts are deterministic. Note for Phase 2: MC-dropout uncertainty would need
  dropout switched on deliberately.

## Speed-ups (Push-T checkpoint, RTX 5060 Laptop 8 GB)

Measured with `scripts/benchmark_dino_wm.py`: one 300-candidate, horizon-5 rollout
from a real Push-T start frame, goal = frame 25 steps later, upstream's cost
(`alpha=1, mode=last`). Reference = unmodified model in fp32 (chunked to fit).
"Elite overlap" = share of CEM's top-30 candidates that stay the same; "update
change" = relative change of the elite-mean action CEM moves to.

| Setup | Time / rollout | Peak VRAM | Latent drift (σ=1 / 0.1 / 0.02) | Elite overlap (σ=1 / 0.1 / 0.02) | Update change |
|---|---|---|---|---|---|
| Original, chunk 50 | 26.2 s | 3.8 GB | — | — | — |
| *Noise floor: original, chunk 25* | — | — | 0.21% / 0.11% / 0.10% | 100 / 100 / 100% | 0% |
| Patches, fp32, chunk 150 | 15.1 s (1.7×) | 3.0 GB | 0.25% / 0.16% / 0.09% | 100 / 100 / 100% | 0% |
| **Patches, TF32, chunk 150 (used)** | **6.9 s (3.8×)** | 3.0 GB | 0.72% / 0.38% / 0.34% | **100 / 100 / 100%** | **0%** |
| Patches, bf16, chunk 150 | 3.6 s (7.2×) | 2.3 GB | 5.3% / 2.6% / 2.7% | 100 / 93 / 83% | up to 0.6% |

σ is the spread of the candidate actions: 1 matches CEM's first iteration, small σ
mimics its last iterations, where candidates are near-identical and rankings are
most fragile. bf16 changes CEM's choices there, so it is not used for reproduction.
Chunk 300 does not fit under the 85% VRAM cap; chunks 50–150 run at similar speed.

## Paper targets (arXiv 2411.04983)

| Env | MPC (Table 1 / Table 8) | Open-loop CEM (Table 8) |
|---|---|---|
| PointMaze | 0.98 | 0.80 |
| Wall | 0.96 | 0.74 |
| Push-T | 0.90 | 0.86 |

Protocol stated in the paper: 50 initial/goal pairs per environment; Push-T goals
reachable within 25 steps (= `goal_H 5` × `frameskip 5`). **Not stated:** the MPC
replanning budget. The repo config (`max_iter: null`) replans until every episode
succeeds, which never terminates if any episode fails; upstream's Slurm jobs had a
12 h timeout (A6000/H100). Our runs therefore cap `max_iter` and report success
after each replanning round.

## Results

| Env | Seed | n_evals | Success | Paper | Wall-clock | Notes |
|---|---|---|---|---|---|---|
| Push-T | 99 | 5 | 4/5 (0.8) | — | 14 min | **Smoke test, unpatched**: `opt_steps=3` (upstream 30), `max_iter=2`. Pipeline verified end to end; not comparable to the paper. Run: `outputs/runs/phase0/pusht/20261002_093428` |
| Push-T | 99 | 5 | 4/5 (0.8) | — | 4 min | **Same smoke test, patched (TF32)**. Episodes 0, 1, 2, 4 identical to the unpatched run to every printed digit; episode 3 diverged at its 3rd CEM step (state dist 27.51 vs 27.36, a near-tie in the elite ranking) and ended at 41.69 vs 37.47, still a success. |

| Push-T | 99 | 50 | **0.90** after MPC round 1; 0.92 after rounds 2 and 3 | 0.90 (MPC) | 2 h 50 min for round 1; ~4.5 h until the crash | **Full upstream settings** (30 CEM steps, 300 samples, top-30, horizon 5), patched runtime (TF32, fused attention, skip-solved, serial simulators). Rounds 4–5 did not complete: OOM while plotting round 4's evaluation (see below; fixed, not rerun). Unsolved (episode indices from 0): after round 1 → 26, 28, 33, 47, 49; after round 3 → 26, 28, 47, 49 (episode 33 was solved in round 2). Regenerate with `scripts/summarize_phase0.py`. Run: `outputs/runs/phase0/pusht/20261010_005403`, log `full_20261010_005356.log` (`full_20261010_003324.log` is the first launch, killed by the simulator-process memory problem) |
| Push-T | 100 | 50 | **0.90** after round 1; **0.94** after rounds 2 and 3 | 0.90 (MPC) | 2 h 50 min for round 1; ~3.5 h total | Same settings as seed 99, capped at 3 rounds; completed (exit code 0). Unsolved after round 3 (0-based): 25, 42, 46. Log `seed100_20261010_053933.log` |

| Push-T | 101 | 50 | **0.82** after round 1; 0.84 after rounds 2 and 3 | 0.90 (MPC) | ~4 h (slowed by a game on the GPU during rounds 1–3) | Same settings, 3 rounds; completed (exit code 0). Unsolved after round 3 (0-based): 0, 2, 9, 10, 19, 29, 30, 38. Log `seed101_20261010_090821.log` |

**Push-T summary over seeds 99, 100, 101** (`scripts/summarize_phase0.py`):

| After MPC round | Seed 99 | Seed 100 | Seed 101 | Mean ± std | Paper |
|---|---|---|---|---|---|
| 1 | 0.90 | 0.90 | 0.82 | 0.873 ± 0.046 | 0.90 |
| 2 | 0.92 | 0.94 | 0.84 | 0.900 ± 0.053 | |
| 3 | 0.92 | 0.94 | 0.84 | 0.900 ± 0.053 | |

Interpretation: the paper's 0.90 lies within one standard deviation of our mean at every
round, so the reproduction is consistent with the paper. Caveats: three seeds is a small
sample (std itself is uncertain); with 50 episodes a rate near 0.9 has a binomial standard
error of about 0.04, so most of the seed-to-seed spread is expected sampling noise from
different start/goal pairs; the paper's replanning budget is unknown, so which round to
compare is a free choice (round 1 gives 0.873, round 3 gives 0.900).

Reading the full run:
- Success per MPC round: 45/50 → 46/50 → 46/50. Round 1 equals the paper's 0.90; later
  rounds add one episode and then none. The paper does not state its replanning
  budget, so 0.90 vs 0.92 depends on how many rounds it allowed. One seed only; the
  pass criterion (within the seed-to-seed spread) needs more seeds.
- Crash: `evaluator.eval_actions` replays the whole action history each round and
  decodes it for the plots. My `decode_obs` patch grouped 10 episodes per batch, which
  is 210 frames in round 4 (21 frames each), and one 1.88 GiB tensor no longer fit under
  the 5.95 GiB cap. Now grouped by frame count (~60 frames per batch); checked at round-5
  size (50 × 26 frames): GPU peak 3.3 GB, output identical to a plain decode (2e-7).
- Operational lessons from this run, all fixed in `dino_wm_patches.py` /
  `plan_dino_wm.py`: 50 forked Push-T simulators (~130 MB each) exhaust the 7.5 GB WSL
  VM → simulators now run serially in-process; upstream encodes the 50 goal images and
  decodes plot frames with autograd on (~3 GB at 50 episodes) → gradient-free; GPU
  overflow spills into system RAM on Windows and hangs WSL → 75% VRAM cap.

**Wall and PointMaze** (upstream configs: 10 CEM steps, 300 samples, top-30, horizon 5; patched
runtime; at most 3 MPC rounds; `scripts/phase0_env_full.sh`):

| Env | Seed | n_evals | After round 1 (one CEM plan) | After replanning | Paper (CEM / MPC) | Notes |
|---|---|---|---|---|---|---|
| Wall | 99 | 50 | 0.54 (27/50) | 0.90 (45/50) after round 2; **0.92 (46/50)** after round 3 | 0.74 / 0.96 | WSL shut down (10 Oct) after round 3's evaluation but before the per-episode summary printed; round 3 is read from the `MPC iter 2 Eval` line (first reported as lost). Log `wall/full_wall_20261010_135226.log` |
| Wall | 100 | 50 | 0.50 (25/50) | 0.92 (46/50) after round 2; **0.96 (48/50)** after round 3 | 0.74 / 0.96 | 3 Nov. Log `wall/wall_seed100_20261103_055240.log` |
| Wall | 101 | 50 | 0.52 (26/50) | 0.92 (46/50) after round 2; **0.96 (48/50)** after round 3 | 0.74 / 0.96 | 3 Nov. WSL shut down right after round 3's evaluation (as for seeds 99 and 100); round 3 read from the eval line. Log `wall/wall_seed101_20261103_120805.log` |
| PointMaze | 99 | 50 | 0.86 (43/50) | **1.00 (50/50)** after round 2 | 0.80 / 0.98 | Completed 3 Nov in ~75 min of CEM time; stopped after round 2 because every episode was solved. Log `point_maze/point_maze_20261103_031258.log` |

PointMaze matches the paper. Wall after replanning is close to it: 0.95 on average after round 3
(0.92, 0.96 and 0.96 over three seeds) against 0.96. Wall's single-plan result is lower than published on
all three seeds (0.54, 0.50 and 0.52 against 0.74), so this gap is consistent rather than seed luck. The
cause is not known yet: we use the upstream Wall config, and the paper does not say exactly how its
CEM row was evaluated, so mapping it to our round 1 may be the mismatch.

Notes from the smoke test:
- Push-T success = block + agent positions within 20 and angle within π/9 of the goal
  (`env/pusht/pusht_wrapper.py`). `state_dist` also includes velocity, so it can stay
  large on a success. MPC counts an episode as solved if it succeeds at any replan.
- Throughput on the RTX 5060 (8 GB): **~2.2 min per CEM step for 5 episodes**
  (300 samples, chunks of 50, GPU at ~7.45 GB / 99%). Upstream full settings
  (50 episodes × 30 steps per replan) ≈ 11 h per replan round. The full protocol
  is not practical locally without speed-ups.
