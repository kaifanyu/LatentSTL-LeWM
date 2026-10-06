# LatentSTL-LeWM

Research code for **Signal Temporal Logic (STL) over images and learned visual
latents**, using LeWorldModel and the LIBERO robot simulator.

The task is to move a Panda arm across a table while preserving a mug. We test
whether visual predicates can recognize that specification, whether learned
dynamics can predict it, and whether it can guide action selection.

<img src="docs/assets/start.png" alt="Initial arm and mug" width="192"> <img src="docs/assets/goal.png" alt="Supplied goal image" width="192">

## Start here

- [Workflow and commands](docs/workflow.md): collect → train → evaluate → plan.
- [Results and limitations](docs/results.md): what the experiments establish.
- [Data format](docs/data.md) and [LeWM architecture](docs/lewm_adapter_notes.md).
- [Local artifacts](docs/artifacts.md): checkpoints, datasets, and retained runs.

## Setup

Tested on Linux with Python 3.11 and 3 × RTX 2080 Ti (11 GB). Training and
inference use **FP16**, 128 × 128 single-camera images, and independent GPU jobs.
There is no DDP. An NVIDIA driver with CUDA/EGL support, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/) are required.

```bash
git clone https://github.com/kaifanyu/LatentSTL-LeWM.git
cd LatentSTL-LeWM
bash scripts/setup.sh
source scripts/env.sh
python scripts/doctor.py
python -m pytest -q
```

Setup installs pinned Python dependencies and clones the pinned LeWM and LIBERO
sources into `third_party/`. It preserves an existing Python 3.11 environment.
For the real simulator replay test:

```bash
RUN_SIMULATOR_TESTS=1 python -m pytest -q tests/test_data.py
```

## Main workflow

```bash
# 1. Collect images/actions with disjoint train/validation/test starts.
python -m latent_stl_lewm.collect --output datasets/libero_mug_v1 \
  --train-episodes 256 --val-episodes 32 --test-episodes 32 \
  --steps 96 --seed 20261004 --render-gpu 0

# 2. Train three independent seeds on GPUs 0, 1, and 2.
python scripts/launch_3gpu.py --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/training --max-steps 20000 --epochs 100
python scripts/status.py runs/training

# 3. Once training finishes, launch the visual-predicate benchmark.
python scripts/launch_visual_campaign.py \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint runs/training/seed_2/train/best.pt \
  --output runs/visual_benchmark
```

Always choose a **new output directory**. The benchmark continues in the
background; inspect `runs/visual_benchmark/status.json` and open its `index.html`
for metrics and replays. The default queue collects 864 trajectories across
eight conditions, evaluates four representations, then audits global-latent
forecasts. For one GPU, use the individual commands in the workflow guide.

The last completed benchmark used a predictor fine-tuned for eight-step
rollouts. Its reproduction command is in [the workflow](docs/workflow.md).
Checkpoints and datasets are **not included in this Git repository**; generate
them locally or use the preserved local artifacts listed in the artifact guide.

## What is being compared?

| Method | Representation used by the predicate |
| --- | --- |
| `global` | One whole-image LeWM latent vector |
| `spatial` | Corresponding LeWM patch features in the mug/goal regions |
| `pixels` | RGB differences in fixed image regions |
| `red_pixels` | Red-color evidence for the mug, RGB differences for the goal |

For each observed trajectory, STL checks **preserve the mug throughout AND
reach the goal sometime**. A separate forecast audit evaluates predicted
global latents. The visual benchmark uses scripted collection trajectories;
offline ranking is not CEM action generation. The existing world model predicts
global latents only, not spatial patch features.

## Current evidence

On 864 fresh trajectories, using validation-balanced thresholds:

| Method | STL / physical-label agreement | Unsafe frames accepted as safe |
| --- | ---: | ---: |
| Global LeWM | 56.6% | 54.1% |
| Spatial LeWM | 91.0% | 16.8% |
| RGB crops | 93.4% | 7.0% |
| Red-mug pixels | 93.3% | 5.0% |

These are classification results, not closed-loop success rates. Small isolated
mug movements expose failures even in the better methods. Earlier successful
CEM trials used demonstration waypoints and action priors; an advantage from
STL and planning from scratch remains unestablished. See the
[full interpretation](docs/results.md).

## Repository layout

```text
latent_stl_lewm/   Models, simulator, predicates, STL, collection, and planners
scripts/          Setup, launchers, monitoring, audits, and report generation
tests/            Automated behavioral and integration tests
docs/             Workflow, architecture, data schema, and compact results
datasets/         Local data only (ignored)
runs/             Local checkpoints, traces, logs, and videos (ignored)
third_party/      Pinned upstream clones, installed by setup (ignored)
```

Experimental planners are retained for reproducibility; the workflow guide
identifies the main entry points. Generated smoke runs and stale launch reports
have been removed. Small result summaries are versioned under `docs/evidence/`.

Built on [LeWorldModel](https://github.com/linzhongzi/le-worldmodel) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Upstream dependencies
retain their own licenses; this repository does not vendor their source trees.
