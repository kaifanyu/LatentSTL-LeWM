# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.14 AS uv
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates ffmpeg git \
        libegl1 libgl1 libgles2 libglib2.0-0 libglvnd0 libglfw3 \
        libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /workspace
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY requirements.lock.txt ./
RUN uv venv --python /usr/local/bin/python .venv \
    && uv pip install --python .venv/bin/python -r requirements.lock.txt

COPY . .
RUN bash scripts/setup.sh \
    && git config --system --add safe.directory /workspace/third_party/le-worldmodel \
    && git config --system --add safe.directory /workspace/third_party/LIBERO
COPY --chmod=755 scripts/docker-entrypoint.sh /usr/local/bin/latentstl-entrypoint

# Also apply to `docker compose exec`, which does not run the entrypoint again.
ENV PATH="/workspace/.venv/bin:${PATH}" \
    PYTHONPATH=/workspace:/workspace/third_party/LIBERO \
    LIBERO_ROOT=/workspace/third_party/LIBERO \
    LIBERO_CONFIG_PATH=/workspace/.cache/libero-docker \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMBA_CACHE_DIR=/workspace/.cache/numba \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# These checks run without GPUs or research artifacts during the image build.
# Import the learned visual backend as well as both executable STL entrypoints.
RUN python -c "from transformers import ViTModel; from latent_stl_lewm.visual_stl import VisualSTLChecker; from latent_stl_lewm.forecast_stl import HistoryConditionedRollout, TimedLatentScorer" \
    && python -m latent_stl_lewm.check_stl --help > /dev/null \
    && python -m latent_stl_lewm.forecast_stl --help > /dev/null

ENTRYPOINT ["latentstl-entrypoint"]
CMD ["bash"]
