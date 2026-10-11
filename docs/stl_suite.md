# Two-GPU STL predicate experiment

This finite experiment starts from an empty Docker workspace and produces data,
two trained models, calibrated image predicates, held-out simulations, forecast
audits, and an automatically refreshed HTML report. The exact frozen plan and
commands are in each run's `protocol.json`; actual commands, GPU assignments,
timestamps, return codes, and log paths are in `status.json`.

## Questions and scope

1. Can each supported visual representation recognize mug preservation and reaching?
2. Do exact temporal formulas distinguish safe reach, collisions, and transient reach?
3. How do results change with arm/mug placement, camera position, illumination,
   trajectory duration, and model training seed?
4. Do global latent forecasts preserve predicate decisions at 1, 2, 4, and 8 steps?

The four supported representations are tested on identical saved images/actions.
This is an RGB monitoring and recorded-action forecast experiment. The action
sequences are scripted simulation behaviors; offline candidate ranking selects
among those recorded behaviors. This run does not establish autonomous CEM
planning or closed-loop task success. Arbitrary STL formulas, an independent
avoid-reference task, and Until are outside this experiment.

## Data, training, and independence

The source collection has 512 training, 64 validation, and 64 test episodes,
each with 96 actions and 97 observations at 20 Hz. Resolution is 128×128 from
`agentview`. Each bank contains eight behaviors restored to the same initial
simulator snapshot; entire banks belong to one split. Collection seed is
20261004. Source scene seeds are 20361004 onward for training, 20461004 onward
for validation, and 20561004 onward for test. Model training consumes only RGB
and raw actions. Physical measurements are separate evaluation diagnostics.

GPU 0 first collects this source dataset. In parallel, GPU 1 collects a small
8/8/8 reference-bootstrap dataset using the exact same collection seed and
simulator settings. It provides the fixed goal image so fresh simulations can
begin before the larger collection finishes. These 24 bootstrap episodes
duplicate source starts, are not additional training or held-out evidence, and
are excluded from result counts. `reference_audit.json` verifies that source and
bootstrap goal/safe images are byte-identical and that fresh scene seeds are
disjoint from all source splits. A failed audit blocks training and evaluation.

Two independent model seeds (0 and 1) train on the same source training split,
one on each GPU: 20,000 optimizer attempts, batch size 32, FP16 AMP, the existing
LeWM architecture, and the existing 5e-5 learning-rate recipe. The best checkpoint
is selected by validation prediction-plus-SIGReg loss. Each model then gets
3,000 steps of frozen-encoder predictor fine-tuning with one-step plus eight-step
loss, batch size 32 and learning rate 3e-5. All validation windows are used.
Fine-tuning verifies the representation hash is unchanged and selects on
validation prediction loss. Both seeds are reported; fresh test performance
never selects a seed or checkpoint.

## Representations and calibration

| Method | Preservation comparison | Goal comparison |
| --- | --- | --- |
| `global` | Cosine distance from initial whole-image LeWM embedding | Cosine distance to fixed goal embedding |
| `spatial` | Spatially corresponding learned patch features in initial mug region | Learned patch features in fixed goal region |
| `pixels` | RGB root-mean-square difference in initial mug region | RGB difference in fixed goal region |
| `red_pixels` | Continuous red-color evidence in initial mug region | Ordinary RGB difference in fixed goal region |

The mug region is detected once from initial RGB using an explicit color/region
prior. It does not follow the mug later. The goal box is fixed at
`(80,32,116,72)` in 128×128 images. Region/color priors are task inputs, so these
comparisons do not isolate encoder architecture alone. The supplied goal image
remains fixed under all scene and appearance changes. Preservation always uses
each evaluated trajectory's own initial image.

For each method and model checkpoint, source **validation frames only** fit two
scalar thresholds. `balanced` chooses balanced-accuracy thresholds;
`false_safe_05` chooses the largest safety threshold accepting at most 5% of
unsafe validation frames. This is a validation operating point, not a promised
5% error rate on fresh data. Goal thresholds and normalization scales are also
fitted on validation. Threshold JSON hashes are frozen before fresh evaluation
and checked before each dependent job. Source test results are diagnostics only.
Pixel baselines repeat identically across seeds and are not independent model
replicates.

## Fresh trial matrix

| Wave | First scene seed | Banks per scenario | Actions | Simulated seconds | Fresh trajectories |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave1 | 951000 | 4 | 96 | 4.8 | 288 |
| wave2 | 952000 | 8 | 96 | 4.8 | 576 |
| wave3_long | 953000 | 12 | 192 | 9.6 | 864 |
| Total | | 24 | | | 1,728 |

All waves contain all eight scenarios below. Scene seeds are paired across
scenarios; each bank's seed is the wave seed plus its bank index. Waves have
disjoint seeds, also disjoint from the 941000-range infrastructure smoke trials.
Each model/method scores the exact same stored trajectory;
simulations are not repeated per model. There are 13,824 trajectory-method-model
evaluations (1,728×4×2), each at two threshold operating points. These are repeated
measurements of 1,728 trajectories, not 27,648 independent simulations.

| Scenario | Change from baseline |
| --- | --- |
| baseline | Default task and camera |
| start_jitter_3cm | Uniform arm-start jitter ±3 cm per axis |
| start_x_plus3cm | Arm start shifted +3 cm on x |
| mug_x_plus3cm | Mug shifted +3 cm on x before settling |
| mug_x_minus3cm | Mug shifted −3 cm on x before settling |
| mug_y_plus3cm | Mug shifted +3 cm on y before settling |
| camera_x_plus2cm | Camera shifted +2 cm on x |
| dim_085 | Saved/runtime RGB brightness multiplied by 0.85 |

Every bank restores exactly the same initial image and simulator state before
each of nine behaviors: `noop`, `low_motion`, `safe_arc`, `collision_straight`,
`near_collision`, `safe_lateral`, `random_smooth`, `random_reach`, and
`reach_leave`. These labels describe scripted intentions; outcomes are measured,
not assumed. Collectors use simulator state to supply diverse behavior coverage.
`reach_leave` performs a safe-arc attempt during the first two thirds, then
returns toward its initial end-effector position.

The 192-action wave is a duration stress test: the script's path timing also
changes with the episode length. Thresholds remain calibrated on 96-action
source episodes. Report 96- and 192-action results separately; this is not a
pure horizon extension with an identical action prefix.

## Predicates, formulas, and measurements

Physical preservation means mug translation ≤2 mm **and** rotation ≤2 degrees
relative to the episode start. Physical reach means end effector within 3 cm of
the fixed goal. These physical labels audit RGB predicates; they are not runtime
predicate inputs. Each visual margin is `(threshold - distance) / validation_std`,
with a 1e-6 minimum scale. Nonnegative margins satisfy a predicate.

All formulas use exact min/max over sampled observations, including the initial
frame. `G preserve` takes the minimum safety margin; `F reach` takes the maximum
goal margin. Their conjunction takes the minimum of those two values. Sustained
reach instead requires ten consecutive satisfying observations somewhere in the
episode, while preserving the mug throughout. Ten observations span 0.45 seconds
between their first and last timestamps at 20 Hz. A past safety violation stays
in the full-episode result even if the mug later recovers.

Reports include frame-level false-safe and false-unsafe rates, balanced accuracy,
trajectory STL/physical agreement, sustained-reach agreement, missed disturbances,
alarm delay, and region-detection fallbacks. Offline ranking compares terminal
goal, eventual goal, weighted penalties, STL, and sustained STL on the same
nine-candidate banks. Per-shard confidence intervals resample entire initial-state
banks. Paired scenarios/methods share seeds and must not be treated as independent
observations. Pooled descriptive rates are not independent-replicate intervals.

The first bank in each shard also supplies static image interventions: x/y shifts
of 0, 1, 2, 4, 8, and 20 mm and yaw changes of 0, 1, 2, 4, 8, and 20 degrees.
These are saved and scored separately from executable trajectories and temporal
formulas. They test whether small physical disturbances are visually observable.

Global forecast audits cover all eight scenarios in wave1 and wave3_long for
both model seeds. The model observes history through each start and predicts
under recorded future actions at horizons 1/2/4/8. Forecast formula windows are
local `[start,start+H]`; they do not assert full-episode satisfaction. The current
predictor forecasts global embeddings only; spatial forecasts are not claimed.

## Execution and inspection

The supervisor assigns at most one experiment process to each GPU. Training seed
0 prefers physical GPU 0, seed 1 GPU 1. Other jobs use whichever GPU becomes free.
CUDA visibility and the simulator EGL device are both set explicitly. Source
collection and reference collection start concurrently; fresh collection can run
while source preparation/training progresses. Dependency failures are retained
in logs and block dependent jobs; independent work continues.

The run executes its copied `source_snapshot`, including source hashes and BDDL
assets. Symlinks use the already installed `/workspace/.venv`, `third_party`, and
`.cache` inside the container. Editing the working repository does not change an
active run. No container restart is needed. Artifacts are persisted under the
host repository's bind-mounted `runs` directory.

After Docker setup, prepare from the host using a **new** name. This executes
preparation inside the image, so no host Python installation is needed:

```bash
cd LatentSTL-LeWM
bash scripts/docker.sh exec python scripts/run_stl_suite.py --prepare \
  --output /workspace/runs/stl_suite_20261011
```

Launch inside the existing container, independently of the SSH shell:

```bash
docker exec -d latentstl-lewm bash -c \
  'exec python /workspace/runs/stl_suite_20261011/source_snapshot/scripts/run_stl_suite.py --output /workspace/runs/stl_suite_20261011 > /workspace/runs/stl_suite_20261011/supervisor.log 2>&1'
```

`index.html` refreshes automatically, and shows progress before metrics exist.
Trial pages provide both operating points, actual replay videos, predicate traces,
physical labels, JSON/HDF5 artifacts, and the applicable commands. The first bank
is a predetermined visualization sample; it is not selected for favorable scores.
`TRIALS.md` in each run is this description frozen at launch. `protocol.json`
is authoritative for smoke versus full settings. Smoke runs use two optimizer
steps and are infrastructure checks, excluded from research conclusions.

Inspect from the host:

```bash
watch -n 2 nvidia-smi
tail -f runs/stl_suite_20261011/supervisor.log
python3 -m json.tool runs/stl_suite_20261011/status.json
```

Serve the report on host loopback with `python3 -m http.server 8000 --bind
127.0.0.1 --directory /home/kaifany/projects/LatentSTL-LeWM/runs`. Forward port
8000 from a local laptop terminal with `ssh -N -L 8000:127.0.0.1:8000
kaifany@YOUR_SSH_HOST`, then open
`http://localhost:8000/stl_suite_20261011/`.

To stop just this suite and its child experiment processes:

```bash
docker exec latentstl-lewm python -c \
  'import os,signal; from pathlib import Path; os.kill(int(Path("/workspace/runs/stl_suite_20261011/supervisor.pid").read_text()), signal.SIGTERM)'
```

The supervisor allows 30 seconds for checkpoint flushing before terminating
remaining workers. Completed artifacts remain. It does not silently resume or
overwrite partial trials; use a new run directory for another suite. A failed
scientific metric is a result, while a nonzero process exit is an execution
failure. Check both the status and the report.
