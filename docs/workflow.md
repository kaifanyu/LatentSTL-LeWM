# Workflow

Run commands from the repository root after `source scripts/env.sh`. Use new
output directories; scientific artifacts are never silently overwritten.

For the current two-RTX-3090 Docker environment, start with the
[Docker setup](docker.md) and [two-GPU STL suite](stl_suite.md). The three-GPU
launchers below reproduce the original workflow and require three visible GPUs.

## 1. Collect the training dataset

```bash
python -m latent_stl_lewm.collect --output datasets/libero_mug_v1 \
  --train-episodes 256 --val-episodes 32 --test-episodes 32 \
  --steps 96 --seed 20261004 --render-gpu 0
python -m latent_stl_lewm.dataset datasets/libero_mug_v1/trajectories.h5
```

Scripts use simulator state to collect coverage. Stored RGB/actions are the
world model's inputs; physical measurements are kept separately. Starts are
split by entire candidate banks. Details: [data format](data.md).

## 2. Train LeWM

```bash
python scripts/launch_3gpu.py --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/training --max-steps 20000 --epochs 100
python scripts/status.py runs/training
```

This launches three seeds, batch size 32, FP16, without DDP. Each seed produces
`train/best.pt`, `train/last.pt`, validation metrics, and held-out evaluation.
The legacy MPC stage skips execution when its evidence gate fails; that is an
experimental finding, not a launcher failure. `best.pt` is selected by validation
prediction-plus-SIGReg loss. `last.pt` is for resuming training.

The initial model trains the encoder and predictor from scratch. It uses the
LeWM architecture; it is not an off-the-shelf pretrained LIBERO checkpoint.

Optional multi-step dynamics fine-tuning freezes the representation and trains
the predictor on one-step plus eight-step autoregressive losses:

```bash
CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.finetune_multistep \
  --checkpoint runs/training/seed_2/train/best.pt \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/multistep --max-steps 3000 --future-steps 8 --batch-size 32 --amp
```

The published local benchmark summary used the original October 4 seed-2 model
followed by this fine-tuning recipe. Fresh retraining is not claimed bit-identical.

## 3. Test predicates before relying on forecasts

For the complete three-GPU benchmark:

```bash
python scripts/launch_visual_campaign.py \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/multistep/best.pt --output runs/visual_benchmark
```

The launcher freezes source hashes and a protocol, calibrates from historical
validation frames, freezes thresholds, collects two fresh waves, audits raw
physical labels, evaluates every method, and runs forecast diagnostics.
`--prepare-only` writes the protocol without starting GPU work. The default is
4 + 8 starting seeds per condition, 9 behaviors, and 8 conditions: 864 episodes.
Change `--seed` for fresh held-out starts; `--banks-per-wave 4 8` sets wave sizes.

Open `runs/visual_benchmark/index.html`, or inspect `status.json`, `summary.json`,
and `logs/`. The supervisor PID is in `supervisor.pid`; send it SIGTERM to stop
the queue and its active workers. It runs a finite queue, not an infinite loop.

To compare one method on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.visual_predicate_benchmark \
  --method spatial --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/multistep/best.pt --output runs/spatial_calibration

CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.collect_visual_benchmark \
  --source datasets/libero_mug_v1/trajectories.h5 \
  --output runs/fresh_baseline --scenario baseline --seed 930000 --banks 4

CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.visual_predicate_benchmark \
  --method spatial --dataset runs/fresh_baseline/trajectories.h5 \
  --checkpoint runs/multistep/best.pt \
  --calibration runs/spatial_calibration/calibration.json \
  --output runs/spatial_fresh
```

Available methods: `global`, `spatial`, `pixels`, `red_pixels`. No model weights
are updated here. Validation physical labels calibrate scalar thresholds;
fresh test labels never tune them. The goal image remains fixed under camera
and brightness variations. Region selection and color priors are explicit task
inputs, so this does not isolate encoder architecture alone.

## 4. Audit predicted futures

```bash
CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.visual_forecast_audit \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/multistep/best.pt \
  --calibration runs/visual_benchmark/calibration/global/calibration.json \
  --output runs/forecast_audit
```

This predicts futures under recorded actions at 1/2/4/8 steps and measures
predicate errors. It does not search for actions. Current LeWM dynamics predict
the global representation; spatial prediction needs a matching new predictor.

## 5. Planning experiments

These entry points remain available, but none establishes robust latent-STL
planning from scratch:

| Entry point | Purpose and limitation |
| --- | --- |
| `latent_stl_lewm.mpc` | Matched CEM objectives after a predicate/dynamics evidence gate |
| `scripts/launch_cem_comparison.py` | Local LIBERO versus official Cube weights; transfer assumptions are recorded |
| `latent_stl_lewm.reference_cem` | CEM with calibrated safe/unsafe reference banks |
| `latent_stl_lewm.waypoint_cem` | Demonstration-assisted local waypoints and CEM action refinement |
| `relative_waypoint_cem`, `inverse_waypoint_cem`, `visual_policy` | Development ablations retained to explain failures and priors |

Use each command's `--help` for its required artifacts. The waypoint planner
requires a checkpoint, reference-predicate artifact, action prior, and training
dataset. Its candidate-selection surrogate uses future safety, waypoint progress,
and action penalties; exact full-trace STL is logged separately.

Physical coordinates are not online CEM inputs. Collection scripts and offline
reference/threshold calibration do use physical labels, which must be disclosed.
Successful waypoint trials are not demonstration-free action discovery.

## Where the implementation lives

- `world_model.py`, `train.py`, `finetune_multistep.py`: encoder and dynamics.
- `simulator.py`, `collect.py`, `dataset.py`, `research_scenarios.py`: data and simulator.
- `logic.py`, `reference_predicates.py`, `visual_predicate_benchmark.py`: predicates/STL.
- `collect_visual_benchmark.py`, `visual_forecast_audit.py`: new data and prediction checks.
- `planner.py`, `reference_cem.py`, `waypoint_cem.py`: action search and execution.
- `scripts/report_*.py`, `scripts/audit_research_execution.py`: analysis and audits.
