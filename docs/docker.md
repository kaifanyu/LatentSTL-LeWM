# Docker on the two-RTX-3090 server

The host supplies the NVIDIA driver and NVIDIA Container Toolkit. The image
supplies Python 3.11, the locked PyTorch 2.7.1/CUDA 11.8 packages, EGL libraries,
and the upstream revisions and patches installed by `scripts/setup.sh`.

The host must have Docker Engine, Docker Compose v2 or newer, an NVIDIA driver,
and NVIDIA Container Toolkit configured for Docker. This Compose file selects
physical GPU indices 0 and 1 and requires the `nvidia` runtime. Confirm the host
setup before building:

```bash
nvidia-smi
docker compose version
docker info --format '{{json .Runtimes}}'
```

Run these commands from the repository root:

```bash
cd LatentSTL-LeWM
bash scripts/docker.sh build
bash scripts/docker.sh up
bash scripts/docker.sh doctor
bash scripts/docker.sh test
bash scripts/docker.sh stl-test
bash scripts/docker.sh replay 0
bash scripts/docker.sh replay 1
bash scripts/docker.sh shell
```

The build downloads several GB of dependencies. `up` starts the persistent
`latentstl-lewm` container with GPU devices 0 and 1. Its idle command is
`sleep infinity`; closing a shell leaves the container running. Each new shell
already has the project environment configured. No host CUDA compiler is needed.

Source, `.venv`, and pinned `third_party` sources are in the image under
`/workspace`. `datasets`, `runs`, and `.cache` are bind-mounted from this
repository, so results persist after stopping or recreating the container.
Changing source requires another `build`, then `up`; Docker caches the Python
dependency layer. The build context excludes local data, checkpoints, caches,
Git metadata, and `.env` files.

The image build checks the learned visual backend imports and both timed-STL
CLI entrypoints without requiring GPUs or datasets. `stl-test` runs the focused
CPU tests for temporal semantics, analytical predicates, visual checking,
validation-only calibration, action-conditioned forecasts, and the CEM adapter.
The full `test` command includes these tests too. `doctor` and `replay` separately
verify GPU execution and simulator rendering; passing CPU tests does not verify
either one.

This server uses Docker user namespace remapping. Compose sets `userns_mode:
host` for this container and the helper passes your host UID/GID, so bind-mounted
outputs remain owned by you. The container runs as your user, without privileged
mode. The image allows Git to inspect its two known upstream repositories even
when image ownership is remapped by the daemon. The entrypoint regenerates
LIBERO's configuration in the mounted cache before starting the container command.
The container uses `.cache/libero-docker/config.yaml`, while host setup uses
`.cache/libero/config.yaml`. This keeps container `/workspace` paths from
overwriting the host's simulator configuration. `scripts/env.sh` preserves an
explicit `LIBERO_CONFIG_PATH`, including the value inherited by Compose `exec`.

The shell may display `I have no name!` because the image has no account entry
for your host's numeric UID. This is expected; `id` still shows the correct
UID/GID and saved files remain owned by your host account.

## Check recorded images and latent forecasts

After `build` and `up`, run the same formula across pixels and learned features
inside the container. The commands below use the retained local dataset and
checkpoint; replace their paths with your own mounted artifacts if necessary.

```bash
bash scripts/docker.sh exec python -m latent_stl_lewm.check_stl \
  --dataset /workspace/datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint /workspace/runs/cem_improved_20261005T064943Z/multistep_training/best.pt \
  --methods pixels red_pixels global spatial \
  --formula /workspace/examples/stl_reach_preserve_5s.json \
  --device cuda:0 --batch-size 32 --max-episodes 3 \
  --output /workspace/runs/docker_stl_observed
```

The output directory must be new. `--max-episodes` limits held-out test episodes;
calibration still uses all validation episodes. The resulting calibration JSON,
predicate traces, and report persist in the host's `runs/docker_stl_observed`.
For a CPU/pixel-only run, omit `--checkpoint`, choose `--methods pixels red_pixels`,
and use `--device cpu`. Physical diagnostics remain a separate comparison and
never enter visual feature extraction.

Use that frozen global calibration to audit a recorded-action forecast:

```bash
bash scripts/docker.sh exec python -m latent_stl_lewm.forecast_stl \
  --dataset /workspace/datasets/libero_mug_v1/trajectories.h5 \
  --checkpoint /workspace/runs/cem_improved_20261005T064943Z/multistep_training/best.pt \
  --calibration-artifact /workspace/runs/docker_stl_observed/calibration.json \
  --episode 000288 --start 16 --horizon 8 --deadline 5 \
  --device cuda:0 --output /workspace/runs/docker_stl_forecast
```

Episode `000288` is a test episode in the example dataset; choose a held-out
episode from your own dataset when using different artifacts. The audit retains
the observed task prefix and the original deadline, then compares predicted
global-latent, observed-latent, and physical predicate traces. Spatial features
can be checked on observed images, but the current world model forecasts global
features only.

Time comes from recorded control indices and the dataset frequency. At 20 Hz,
96 actions cover **4.8 seconds**, so a safe prefix cannot establish preservation
through a five-second deadline. The checker reports unresolved requirements as
`pending`; it never stretches the timestamps. Eight predicted actions cover
0.4 seconds, not five seconds. See the [timed STL guide](stl_checking.md) for
formula JSON, holding semantics, calibration reuse, and direct image/latent APIs.

## Complete two-GPU experiment

Use the [STL suite guide](stl_suite.md) for the documented two-GPU queue, its
trial matrix, report server, and graceful stopping. From the host:

```bash
bash scripts/docker.sh exec python scripts/run_stl_suite.py --prepare \
  --output /workspace/runs/stl_suite_v1
docker exec -d latentstl-lewm bash -c \
  'exec python /workspace/runs/stl_suite_v1/source_snapshot/scripts/run_stl_suite.py --output /workspace/runs/stl_suite_v1 > /workspace/runs/stl_suite_v1/supervisor.log 2>&1'
```

Add `--smoke` to preparation for a small end-to-end check. Preparation copies
the code and protocol into the new run directory; the detached supervisor uses
that snapshot. Closing SSH leaves the queue running. Neither `build` nor `up`
starts experiments automatically.

## Collect and train

Inside the container:

```bash
python -m latent_stl_lewm.collect --output datasets/libero_mug_v1 \
  --train-episodes 256 --val-episodes 32 --test-episodes 32 \
  --steps 96 --seed 20261004 --render-gpu 0
```

Open two shells using `bash scripts/docker.sh shell` in two host terminals.
Run one independent training seed in each shell:

```bash
# First shell: GPU 0.
CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.train \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/training/seed_0 --device cuda --seed 0 --batch-size 32 --amp

# Second shell: GPU 1.
CUDA_VISIBLE_DEVICES=1 python -m latent_stl_lewm.train \
  --dataset datasets/libero_mug_v1/trajectories.h5 \
  --output runs/training/seed_1 --device cuda --seed 1 --batch-size 32 --amp
```

Each output directory must be new. CUDA renumbers the single visible GPU to
`cuda:0` within each masked process. The project has no DDP; displaying both
GPUs does not split one training job across them. CUDA visibility does not select
MuJoCo's EGL device: pass `--render-gpu 0` or `--render-gpu 1` for simulator jobs.
Several legacy simulator commands default to device 2, and
`launch_3gpu.py` / `launch_visual_campaign.py` assume three GPUs. Use the new
two-GPU suite or the individual commands above on this server.

For a training memory benchmark before changing batch size:

```bash
CUDA_VISIBLE_DEVICES=0 python -m latent_stl_lewm.train --benchmark \
  --output runs/memory_3090 --benchmark-batches 32 64 128 256 --amp
```

Monitor the hardware from a host terminal with `watch -n 1 nvidia-smi`.
Use `bash scripts/docker.sh stop` after your experiments. It stops all processes
in this container and keeps the mounted files.
