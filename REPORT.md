# Initial LeWM latent-STL run report

Date: October 4, 2026. This report distinguishes software verification from
scientific evidence. The longer training/evaluation campaign is the first
experiment; no closed-loop advantage has been established yet.

## Architecture and environment

The independent environment and source copies are under
`/data3/kaifany/LatentSTL-LeWM`. Existing ImageWAM/LatentSTL artifacts were not
modified. LeWM source revision is
`ca231f9f9d9ab041034b6d05e90b6e04bd6cff82`; LIBERO is
`8f1084e3132a39270c3a13ebe37270a43ece2a01`, with local packaging/PyTorch-load
compatibility changes documented in `scripts/setup.sh`.

The model has 18,032,144 parameters: ViT-tiny encoder, projected 192-dimensional
global CLS embedding, action embedder, and the upstream autoregressive predictor.
It trains jointly with prediction MSE plus 0.09 SIGReg. Images are 128 x 128 with
patch size 16. Three context actions/four images form each training window.
Network operations use FP16 AMP; prediction loss and SIGReg use FP32.

GPU verification succeeded on all three 11-GB RTX 2080 Ti devices. Independent
single-GPU runs use seeds 0, 1, and 2, without DDP.

| Batch | Peak allocated GiB | Peak reserved GiB | Successful updates |
|---:|---:|---:|---:|
| 32 | 1.085 | 1.227 | 5/5 |
| 64 | 1.915 | 2.127 | 5/5 |
| 128 | 3.568 | 3.936 | 5/5 |

These measurements include full forward, backward, and AdamW steps on real RGB
training windows. The initial campaign uses batch 32. Hardware/dependency details
are in `runs/environment_verification.json`; benchmark data are in
`runs/benchmark_20261004_01/benchmark.json`.

## Data and controls

The collection contains a living-room mug obstacle and scripted reaching/motion
coverage. It uses raw executed 7-D OSC_POSE actions at 20 Hz and single-camera RGB.
Collection completed with 256 training, 32 validation, and 32 test trajectories, each
with 96 actions: 30,720 transitions total. Each eight-candidate bank shares an
exact starting snapshot. Bank/scene seeds are disjoint across splits.

Physical measurements are diagnostics only. No physical pose is passed into
the world model or CEM scorer. Scripted data collection uses geometric targets
to create coverage; the learned planner receives image references instead.
The simulator snapshot includes integration, controller, and observation state.
Three repeated identical-action replays matched simulator state and RGB exactly.
Evidence: `docs/simulator_replay_audit.json`.

Collection took 462 seconds. Whole-trajectory physical outcomes were 100 safe
reaches, 42 unsafe reaches, 147 safe misses, and 31 unsafe misses. These labels
describe collection coverage, not learned-model performance. Final dataset
SHA256 is `a59e65b2f130e23acb06da40ab19fcf7a1c54cf15896f67f627c95a8c39ccc35`.

The early `data/smoke_v1` fixture predates the snapshot repair and is used only
for software/memory testing. Scientific results must use the finalized
`datasets/libero_mug_v1/trajectories.h5`.

## Verification

- CUDA FP16 matrix operations on each GPU and package compatibility passed.
- Actual EGL rendering, deterministic replay, dataset alignment/splits and
  diagnostic-input exclusion passed.
- Forty real-image training updates were finite. Checkpoint reload and genuine
  optimizer/scaler/RNG/sampler resume from step 30 to 40 passed.
- Exact/smooth STL, inclusive windows, signs, masks, gradients, action-conditioned
  rollout, toy CEM direction/bounds, matched budgets, and evidence gates passed.
- A supervised train/evaluate job completed end to end. MPC correctly declined
  the unready smoke checkpoint. This is expected and is not a method result.

All 22 tests passed, including real simulator integration. Full output:
`docs/test_results.txt`. Model integration details:
`docs/lewm_adapter_notes.md`.

## Main campaign and pending evidence

Launched at 07:36 UTC on October 4, 2026:
`runs/campaign_20261004T073606Z`. Source code at launch is local Git commit
`c9d734f`; the campaign additionally preserves a source archive and data hash.

| GPU | Seed | Supervisor PID | Training PID |
|---:|---:|---:|---:|
| 0 | 0 | 748507 | 748511 |
| 1 | 1 | 748508 | 748512 |
| 2 | 2 | 748509 | 748510 |

These are detached processes and continue after the terminal/session ends.
Use `python scripts/status.py` for current progress, since the table records
launch PIDs rather than a live status. Logs and results are in the corresponding
`seed_0`, `seed_1`, and `seed_2` directories.

The launch budget is 20,000 steps or 100 epochs per seed, whichever occurs first.
Full validation runs every 500 steps; `best.pt` minimizes validation total loss.
Checkpoints/logs/configs/source snapshots are kept in the timestamped campaign.
Each worker automatically runs held-out evaluation, then the gated MPC runner.

Evaluation measures full real-trace predicate validity; 1/2/4/8-step prediction
errors, robustness correlations/sign errors, persistence/shuffled-action baselines;
and same-start candidate ranking for terminal, eventual, and STL objectives.
The MPC stage runs only if earlier evidence passes its documented engineering
checks. Otherwise it records the reasons for skipping.

Prediction horizon, robustness correlation, ranking quality, and closed-loop
results are **pending training and evaluation**. They must not be inferred from
training-loss changes or the smoke tests.

## Known limitations and next decisions

- The first preservation predicate uses a global embedding and may penalize
  desired arm motion. Level 0 must diagnose that before interpreting dynamics.
- Initial cosine thresholds (goal 0.02, preservation 0.01) are uncalibrated and
  are not fitted to held-out physical labels.
- Eight raw steps span 0.4 seconds. Prefix ranking is not evidence for completing
  a full reaching task; longer horizons need separate agreement measurements.
- The dataset covers one scene and a small set of policies/start perturbations.
  Three training seeds do not establish broad scene generalization.
- CEM bounds are restricted to training-observed action extrema intersected with
  environment bounds. This does not guarantee every action sequence is in the
  training distribution.
- Frozen-encoder support presently uses a trained local LeWM checkpoint. External
  V-JEPA integration, multiview dynamics, and language grounding remain later work.
