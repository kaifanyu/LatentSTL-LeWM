# Check timed STL on recorded robot trajectories

The new checker evaluates the same explicit formula over pixel, global LeWM, or
spatial LeWM predicate traces. It also evaluates a separate physical diagnostic
oracle. This is recorded-trajectory monitoring; it does not train a model, run a
robot, or establish that a latent dynamics predictor forecasts these predicates.

## Docker and suite interoperability

After [Docker setup](docker.md), run this checker as a separate command. The
[two-GPU suite](stl_suite.md) retains its existing finite-episode benchmark and
local-window forecast protocol; it does not invoke these seconds-based monitors.
Both workflows use the same installed LeWM/LIBERO sources. Checkpoints are local
files, not model-hub downloads; choose a completed model from the mounted `runs`
directory.

For example, after the suite's source collection and seed-0 fine-tuning finish,
check its source dataset with the trained model:

```bash
bash scripts/docker.sh exec python -m latent_stl_lewm.check_stl \
  --dataset /workspace/runs/stl_suite_v1/dataset/trajectories.h5 \
  --checkpoint /workspace/runs/stl_suite_v1/models/seed_0/finetune/best.pt \
  --methods pixels red_pixels global spatial --device cuda \
  --formula /workspace/examples/stl_reach_preserve_5s.json \
  --output /workspace/runs/timed_check_seed0
```

Replace `stl_suite_v1` with the actual suite directory and use a new output
directory. This command calibrates on that source dataset's validation split and
evaluates its test split. Its 96-action recordings cover 4.8 seconds, so a
five-second formula can remain pending. Use an output under `/workspace/runs`
for host persistence; `/tmp` examples later in this guide are disposable and
disappear when the container is recreated. Historical checkpoint paths in those
examples refer to retained local artifacts and are absent from a fresh checkout.

The canonical CLI requires a complete HDF5 dataset with top-level
`control_freq_hz`, per-episode integer `control_step` indices aligned with RGB,
an explicit goal image, and diagnostic labels for its separate calibration and
oracle comparison. Fresh calibration needs both validation and test episodes.
Reusing its `calibration.json` requires the exact same dataset hash and matching
checkpoint/predicate definitions. The forecast CLI uses that same artifact.

The suite's fresh stress-test shards contain **test episodes only**, and its
per-method benchmark calibration JSON has a different schema. Those files are
not direct substitutes for this CLI's source dataset/calibration pair. New
shards record explicit control/frame frequencies and action repetition; older
shards may omit top-level `control_freq_hz` and are rejected rather than assigned
an inferred clock. Existing datasets and frozen calibration are never rewritten
by the checker. To evaluate a custom formula on such traces, use the direct
`VisualSTLChecker` API with explicitly supplied timestamps, source-calibrated
thresholds, and references, retaining the source/evaluation provenance; do not
fit thresholds on stress-test labels.

## Run

From the repository root, after setting up the existing environment:

```bash
source scripts/env.sh
python -m latent_stl_lewm.check_stl \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --methods pixels red_pixels \
  --formula examples/stl_reach_preserve_5s.json \
  --output /tmp/stl_pixels_example \
  --max-episodes 3
```

Use a new output directory for every invocation. `--max-episodes` limits test
episodes only. Validation calibration still uses the complete validation split;
`--max-calibration-episodes N` can shorten a smoke run, but both safe/unsafe and
reached/not-reached classes must remain represented. A small smoke run is not a
performance estimate.

To compare learned representations as well:

```bash
python -m latent_stl_lewm.check_stl \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/cem_improved_20261005T064943Z/multistep_training/best.pt \
  --methods pixels red_pixels global spatial \
  --formula examples/stl_reach_hold_complete_5s.json \
  --device cpu --batch-size 32 \
  --output /tmp/stl_all_backends_example --max-episodes 3
```

CPU inference works but learned feature extraction can be slow. Select CUDA when
available in the environment. No checkpoint is needed for pixels/red_pixels.

Without `--formula`, `--deadline 5 --hold 1` builds preservation through five
seconds and a one-second goal hold *completed* by five seconds:

\[
\mathbf G_{[0,5]}p_{\rm preserve}
\land \mathbf F_{[0,4]}\mathbf G_{[0,1]}p_{\rm reach}.
\]

In contrast, \(\mathbf F_{[0,5]}\mathbf G_{[0,1]}p_{\rm reach}\) allows the
hold to begin at five seconds and finish at six seconds.

## Time and incomplete traces

Time comes from each episode's stored `control_step / control_freq_hz`, relative
to its first observation. The first stored frame is treated as task start, even
if its recorded control index is nonzero; retain the original prefix when the
deadline belongs to an earlier task start. Missing or invalid time metadata is an error. At
20 Hz, 96 actions produce 97 observations covering **4.8 seconds**. These are
never stretched to five seconds. Five seconds needs 100 transitions and 101
inclusive observations. If an action is repeated, use the actual recorded control
indices; do not multiply stored raw-step timestamps by the repeat factor again.

`timed_stl.py` uses inclusive intervals in seconds and evaluates nested operators
at relative times. Until requires its left operand before, but not necessarily
at, its right-operand witness. The monitor returns lower/upper robustness bounds
for missing future data. Status is `satisfied`, `violated`, `pending`, or
`boundary` for an exact zero margin. A partial trace can already establish a
violation or a successful Eventually witness; `complete` separately records
whether its entire formula horizon is covered. A safe 4.8-second prefix cannot
establish Always-preservation through five seconds. A decided status can coexist
with `complete=false` and `robustness=null`: its sign is fixed, while future
samples can still change the numerical robustness.

This is sampled-time STL. Events between frames are unobserved, and an absence
of sampled violations does not establish continuous-time safety.

## Predicates and calibration

The CLI provides canonical atoms `reach`, `preserve`, and `safe` (an alias of
`preserve`). Their margins are `(threshold - distance) / positive_scale`.
Reaching compares against `references/goal_rgb`; preservation compares against
each episode's initial frame. Methods follow the existing visual benchmark:

| Method | Preservation | Reaching |
| --- | --- | --- |
| `pixels` | RGB RMS difference in initial mug ROI | RGB RMS difference in fixed goal ROI |
| `red_pixels` | Red-color evidence difference in initial mug ROI | RGB goal ROI |
| `global` | Whole-image projected LeWM latent cosine distance | Whole-image goal latent cosine distance |
| `spatial` | Corresponding LeWM patch cosine distances, mug ROI weights | Corresponding patch distances, goal ROI weights |

The canonical benchmark uses 128x128 RGB images, fixed goal ROI, and a mug ROI
located once from the initial image. Those scene assumptions are written into
the artifact; they are not general-purpose object detection. The global method
ignores ROIs. Existing global world-model rollout cannot supply spatial tokens.

Validation physical labels supervise calibration using the existing benchmark's
`balanced` or `false_safe_05` operating point. The latter is an empirical
validation budget, not a guarantee on test data. Test labels never change the
thresholds, reference extraction, or feature encoder. Physical margins use the
dataset's declared translation/rotation/goal tolerances when all corresponding
diagnostics are present; otherwise the oracle clearly identifies its +/-1
recorded-label encoding. Visual and physical robustness magnitudes are not
directly compared.

For another formula with the same dataset/model, reuse the frozen artifact:

```bash
python -m latent_stl_lewm.check_stl \
  --dataset datasets/libero_mug_v1/trajectories.h5 --methods pixels red_pixels \
  --calibration-artifact /tmp/stl_pixels_example/calibration.json \
  --formula examples/stl_preserve_until_reach_5s.json \
  --output /tmp/stl_until_example --max-episodes 3
```

Artifacts must match dataset and, for learned methods, checkpoint SHA-256 hashes.
Old benchmark threshold-only JSON files are not accepted as provenance-complete
artifacts. Recalibrating on a different dataset must be an explicit new run.

## Outputs and custom formulas

- `calibration.json`: validation episode IDs, thresholds, scales, predicate
  definitions, dataset/model hashes; written before test feature extraction.
- `formula.json`: the exact evaluated formula.
- `report.json`: per-backend statuses and agreement on pairs whose visual and
  physical statuses are both decided. Pending and zero-boundary counts stay
  visible; they are excluded from agreement denominators.
- `traces/episode_*.json`: timestamps, predicate margins, ROI selection, physical
  oracle and visual monitor bounds. Infinities serialize as `+inf`/`-inf` strings.

Read decided agreement together with all status counts. On truncated traces,
decided comparisons can contain only violations; high agreement on that subset
does not establish goal completion or accurate classification of all episodes.

Formula JSON supports predicates, `true`, `false`, `not`, `and`, `or`, `implies`,
`iff`, `always`, `eventually`, `until`, `release`, and `weak_until`, including
nested operators. The examples show argument structure. Use `interval: [a,b]`
in seconds. Past-time and Next operators are not part of this monitor.

For custom object predicates, explicit thresholds, named references, regions,
and feature-level use, use `VisualPredicateSpec` and `VisualSTLChecker` from
`latent_stl_lewm.visual_stl`. Define what the visual atom means before composing
its timed formula. The canonical CLI deliberately rejects unknown atom names;
it does not fabricate physical labels for a custom predicate such as `grasped`.

## Check action-conditioned LeWM forecasts

The separate forecast audit uses real observations/actions up to a selected
origin, then predicts a candidate sequence of recorded actions. Actual future
images are encoded only after prediction, for independent comparison. The
original task start and observed predicate history remain part of the formula.

```bash
python -m latent_stl_lewm.forecast_stl \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/cem_improved_20261005T064943Z/multistep_training/best.pt \
  --calibration-artifact /tmp/stl_all_backends_example/calibration.json \
  --episode 000288 --start 16 --horizon 8 --deadline 5 \
  --device cpu --output /tmp/stl_forecast_example
```

The output compares predicted, observed-latent, and physical predicate traces
and monitor bounds. Eight transitions cover 0.4 seconds at 20 Hz; the audit
does not call that a five-second prediction. Only global dynamics is available.
The spatial checker works on observed patch features, not imaginary patch
rollouts. Neither a positive forecast score nor a successful code test verifies
the predictive accuracy of the trained world model.

For candidate planning, `TimedLatentScorer.evaluate(rollout)` returns bounds even
when the forecast does not cover the formula. The callable planner adapter
requires exact robustness and raises on unresolved bounds; it does not invent
a terminal value for missing future. A terminal critic or longer credible
forecast is a separate modeling decision.

```python
from latent_stl_lewm.forecast_stl import HistoryConditionedRollout, TimedLatentScorer
from latent_stl_lewm.planner import CEMConfig, plan_cem

# observed_z[T,D] are RAW model.encode outputs; times[T] are actual seconds.
# observed_actions[T-1,A] excludes the next proposed action.
# checker is a global VisualSTLChecker with explicit calibrated specs.
scorer = TimedLatentScorer(
    checker, formula, observed_z, times,
    references={"initial": initial_z, "goal": goal_z}, step_seconds=0.05,
)
dynamics = HistoryConditionedRollout(model, observed_z, observed_actions)
plan = plan_cem(
    dynamics, observed_z[-1: ], goal_z, initial_z, action_low, action_high,
    config=CEMConfig(horizon=100), trajectory_scorer=scorer,
)
```

The dynamics adapter retains observation/action context; the scorer independently
retains specification history. Set the horizon based on the residual deadline
and validated model accuracy; `100` above demonstrates time coverage only.
The main `plan_cem` legacy objectives remain available without a custom scorer.

## Direct image and latent APIs

```python
from latent_stl_lewm.visual_stl import VisualSTLChecker, canonical_reach_preserve_specs

# thresholds is artifact["methods"][method]["thresholds"].
specs = canonical_reach_preserve_specs(thresholds, method=method, initial_rgb=rgb[0])
checker = VisualSTLChecker(
    method, specs, references={"initial": rgb[0], "goal": goal_rgb},
    checkpoint=checkpoint, device="cpu",
)
checked = checker.check_rgb(rgb, timestamps, formula)
print(checked["result"].status, checked["result"].lower, checked["result"].upper)

# Encoded or predicted GLOBAL features can be supplied without images.
# Use a global checker and matching global calibration for this path.
checked = global_checker.check_features(
    predicted_z, prediction_times, formula,
    references={"initial": initial_z, "goal": goal_z}, source="predicted_global_latents",
)
```

Use `TimedLatentScorer` when combining an existing observed prefix with future
predictions. Calling `check_features` on only a suffix explicitly evaluates a
new trace; it does not recover omitted history. `STLMonitor.append` is the
incremental observed-trace API and retains one task start across updates.

`predicate_library.py` provides analytical margins for scalar/vector ranges,
position, separation, polytope membership, quaternion orientation, speed/norm
limits, and externally supplied collision clearance, plus an extensible
`AtomicPredicate` registry. `physical_mug_predicates` exposes separate normalized
translation, rotation, and reach margins for diagnostics. This is not a claim
that image similarity independently estimates translation or rotation in metric
units; new perceptual heads must establish that grounding before using those
physical meanings at runtime.

## Local integration results

The October 10 integration check used the retained multi-step LeWM checkpoint
and original dataset: all 32 validation episodes calibrated the predicates,
then all 32 test episodes were checked without changing thresholds.

For preservation plus a 0.5-second goal hold completed within the recorded
4.8 seconds, the image/physical formula agreement was 29/32 for pixels,
red-pixel preservation, and spatial LeWM; global LeWM agreement was 22/32.
Global LeWM accepted one physically failing episode. The other three methods
rejected three physically successful episodes. This is an observed-trajectory
check on this split, not a closed-loop success result or tolerance guarantee.

For the five-second reach/preserve formula, no episode had complete temporal
coverage. The oracle reported 9 violated and 23 pending; no successful
five-second certificate was fabricated from a 4.8-second recording. A separate
eight-step forecast audit preserved the original task clock and reported
predicted, observed-latent, and physical statuses independently.

Compact results are in [evidence/timed_stl_checking.json](evidence/timed_stl_checking.json).
Full local artifacts are under `runs/timed_stl_20261010_five_seconds/`,
`runs/timed_stl_20261010_reach_hold/`, and `runs/timed_stl_20261010_forecast/`.
