#!/usr/bin/env bash
set -euo pipefail
LEWM_DOCKER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LEWM_DOCKER_ROOT"
export LEWM_UID="$(id -u)" LEWM_GID="$(id -g)"

case "${1:-help}" in
  build)
    docker compose build lewm
    ;;
  up)
    mkdir -p datasets runs .cache
    docker compose up -d lewm
    ;;
  shell)
    docker compose exec lewm bash
    ;;
  doctor)
    docker compose exec -T lewm python scripts/doctor.py --expected-gpus 2
    ;;
  test)
    docker compose exec -T lewm python -m pytest -q
    ;;
  replay)
    docker compose exec -T -e RUN_SIMULATOR_TESTS=1 \
      -e "LEWM_RENDER_GPU=${2:-0}" lewm python -m pytest -q tests/test_data.py
    ;;
  exec)
    shift
    docker compose exec -T lewm "$@"
    ;;
  logs)
    docker compose logs --tail 100 lewm
    ;;
  stop)
    docker compose stop lewm
    ;;
  config)
    docker compose config
    ;;
  help|--help|-h)
    cat <<'HELP'
Usage: bash scripts/docker.sh COMMAND
  build       Build the pinned Python/CUDA project image
  up          Create or start the persistent container with both GPUs
  shell       Open an interactive shell inside the container
  doctor      Check CUDA/FP16 on both GPUs and pinned upstream sources
  test        Run the existing test suite (simulator test is opt-in)
  replay [0|1]  Run the real EGL/MuJoCo replay test on the chosen GPU
  exec ...    Run a command in the container (without a terminal)
  logs        Show recent container startup logs
  stop        Stop the container; datasets, runs, and caches persist
  config      Show the resolved Docker Compose configuration
After changing source files: build, then up to recreate the container.
HELP
    ;;
  *)
    echo "Unknown command: $1. Use bash scripts/docker.sh help." >&2
    exit 2
    ;;
esac
