# Latent STL with LeWorldModel

Isolated LIBERO experiment workspace for `/data3/kaifany/latent_stl_world_model_task.md`.
Existing `LatentSTL` and `ImageWAM` source and evidence are preserved.

The initial model trains the upstream LeWM architecture locally on a mug
preservation/reach scene. Training uses one 128 x 128 agent camera and raw
7-dimensional OSC_POSE actions at 20 Hz. The adapter has **18,032,144 parameters**.
Its projected CLS latent is global; object-specific spatial preservation is an
explicit limitation of this initial configuration.

## Environment and verification

```bash
cd /data3/kaifany/LatentSTL-LeWM
source scripts/env.sh
python scripts/doctor.py
python -m pytest -q tests
# Includes actual EGL rendering and deterministic simulator replay:
RUN_SIMULATOR_TESTS=1 python -m pytest -q tests/test_data.py
```

The independent `.venv` uses Python 3.11, PyTorch 2.7.1 CUDA 11.8,
Transformers 4.51.3, MuJoCo 3.3.2, and robosuite 1.4.0.
`bash scripts/setup.sh` recreates it from the pinned dependency lock and source
revisions. The first setup needs `uv`, Git, and the system EGL/NVIDIA libraries.

FP16 AMP is used on the RTX 2080 Ti, with FP32 prediction loss and SIGReg.
Actual forward/backward/AdamW probes passed for batch sizes 32, 64, and 128;
peak allocated memory was respectively 1.09, 1.92, and 3.57 GiB. The initial
campaign keeps batch 32 for all three seeds.

## Collect new data

Every collection must use a new output directory:

```bash
python -m latent_stl_lewm.collect --output datasets/libero_mug_v2 \
  --train-episodes 256 --val-episodes 32 --test-episodes 32 \
  --steps 96 --seed 20261004 --render-gpu 2
```

This collects 30,720 transitions, 96 actions plus 97 RGB observations per episode.
Eight policies cover no-op, low motion, reaching, detours, near collisions,
collisions, and random action coverage. Eight candidates share each exact
start snapshot; every bank belongs to one split. There are 32 training banks,
4 validation banks, and 4 test banks. Test/validation scene seeds are disjoint
from training. Physical diagnostic arrays never enter model training or CEM.

Completed data appear as `trajectories.h5`; collection uses a temporary file
until it finishes. Simulator restore includes MuJoCo integration state and
controller/observation state, verified by repeated identical-action replay.

```bash
python -m latent_stl_lewm.dataset datasets/libero_mug_v1/trajectories.h5 \
  --output docs/data_audit.json --videos-dir runs/data_examples_v1
```

## Train three independent seeds

```bash
python scripts/launch_3gpu.py \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --max-steps 20000 --epochs 100
python scripts/status.py
```

GPU 0 runs seed 0, GPU 1 seed 1, and GPU 2 seed 2. There is no DDP. Each
independent worker is detached from the terminal and records its PID, exact
commands, dataset hash, source snapshot, logs, model config, metrics, and
checkpoints under a new `runs/campaign_<UTC>/seed_<N>/` directory. Each seed
stops at 20,000 training steps or 100 epochs, whichever comes first. Validation
covers the full validation split. `best.pt` is selected using the validation
prediction-plus-SIGReg loss, not prediction error alone.

Every 500 steps, the trainer validates and writes a resumable `last.pt`.
After training, its worker evaluates `best.pt` automatically on held-out test
trajectories and writes `evaluation/REPORT.md`, JSON measurements, and plots.
The worker then invokes matched CEM MPC for all three objectives. This stage
writes a skip report unless the earlier evidence gate passes; only a validated
model proceeds to closed-loop simulation.

Monitor:

```bash
python scripts/status.py runs/campaign_<UTC>
tail -f runs/campaign_<UTC>/seed_0/training.log
nvidia-smi
```

To stop a seed gracefully, send `SIGTERM` to its supervisor PID from
`launch.json`; the trainer saves its latest state. Resume training directly
with the original arguments and `--resume <run>/train/last.pt`.

## Evaluate and plan

```bash
python -m latent_stl_lewm.evaluate \
  --checkpoint runs/campaign_<UTC>/seed_0/train/best.pt \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/evaluation_v2 --device cuda \
  --horizons 1 2 4 8 --ranking-horizon 8

python -m latent_stl_lewm.mpc \
  --checkpoint runs/campaign_<UTC>/seed_0/train/best.pt \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --evidence runs/evaluation_v2/report.json \
  --output runs/mpc_v1 --device cuda --render-gpu 2 \
  --objective all --horizon 8 --execute-horizon 1
```

The evaluator reports full real-trace predicate validation, latent prediction
and STL robustness agreement at 1/2/4/8 steps, persistence/shuffled-action
baselines, and matched same-start candidate ranking. Initial explicit cosine
thresholds are goal 0.02 and preserve 0.01; they are uncalibrated experiment
parameters, and no physical labels are used to tune them. Eight raw steps
span only 0.4 seconds, so prefix ranking is not full-task reaching evidence.

MPC refuses mismatched or unready evaluation evidence. Passing software tests
or reducing training loss does not establish that latent STL works. See
[experiment design](docs/design.md) and the evolving [run report](REPORT.md).

## Modules

- `logic.py`: distances, image-reference predicates, exact/smooth finite STL.
- `world_model.py`: upstream LeWM adapter and checkpoint loading; an optional
  frozen encoder backend requires a trained encoder checkpoint.
- `simulator.py`, `collect.py`, `dataset.py`: simulator, isolated dataset,
  leakage-resistant windows and audits.
- `train.py`: single-GPU FP16 training, memory probes, deterministic resume.
- `evaluate.py`: Levels 0–2 metrics, failure diagnosis, plots, MPC evidence gate.
- `planner.py`, `mpc.py`: bounded CEM and receding-horizon execution with matched
  terminal-distance, eventual-reach, and latent-STL objectives.

Source: [LeWorldModel repository](https://github.com/linzhongzi/le-worldmodel),
[paper](https://arxiv.org/abs/2603.19312). Upstream model source is unmodified.
