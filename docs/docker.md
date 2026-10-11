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

This server uses Docker user namespace remapping. Compose sets `userns_mode:
host` for this container and the helper passes your host UID/GID, so bind-mounted
outputs remain owned by you. The container runs as your user, without privileged
mode. The image allows Git to inspect its two known upstream repositories even
when image ownership is remapped by the daemon. The entrypoint regenerates
LIBERO's configuration in the mounted cache before starting the container command.

The shell may display `I have no name!` because the image has no account entry
for your host's numeric UID. This is expected; `id` still shows the correct
UID/GID and saved files remain owned by your host account.

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
