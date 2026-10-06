"""Prepare and launch the reproducible visual-STL benchmark on three GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_stl_lewm.evaluate import sha256

SCENARIOS = ["baseline", "start_jitter_3cm", "start_x_plus3cm", "mug_x_plus3cm",
             "mug_x_minus3cm", "mug_y_plus3cm", "camera_x_plus2cm", "dim_085"]
METHODS = ["global", "spatial", "pixels", "red_pixels"]


def prepare(args):
    import h5py
    checkpoint = args.checkpoint.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    if not args.banks_per_wave or any(n < 1 or n > 1000 for n in args.banks_per_wave):
        raise ValueError("Each wave needs 1–1000 banks to keep seed ranges disjoint")
    waves = [{"name": f"wave{i+1}", "seed": args.seed + 1000*i, "banks": n}
             for i, n in enumerate(args.banks_per_wave)]
    seeds = {wave["seed"] + bank for wave in waves for bank in range(wave["banks"])}
    if min(seeds) < 0 or max(seeds) >= 2**32:
        raise ValueError("Scene seeds must fit unsigned 32-bit integers")
    with h5py.File(dataset, "r") as h:
        if not h.attrs.get("complete", False):
            raise ValueError("Dataset is incomplete")
        splits = {str(ep.attrs["split"]) for ep in h["trajectories"].values()}
        if not {"train", "val", "test"} <= splits:
            raise ValueError("Dataset needs train, val, and test splits")
        existing = {int(ep.attrs["scene_seed"]) for ep in h["trajectories"].values()}
        if seeds & existing:
            raise ValueError("Fresh evaluation seeds overlap the source dataset; change --seed")
    if not args.prepare_only:
        import torch
        if torch.cuda.device_count() < 3:
            raise ValueError("This launcher needs three visible GPUs; use individual benchmark commands for one GPU")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "runs" / f"visual_stl_benchmark_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "source_snapshot", "calibration", "data", "evaluations", "forecasts"):
        (output / name).mkdir()
    files = [p for folder in ("latent_stl_lewm", "scripts", "tests")
             for p in sorted((ROOT / folder).rglob("*.py")) if "__pycache__" not in p.parts]
    files += [ROOT / name for name in ("pyproject.toml", "requirements.lock.txt",
              "latent_stl_lewm/assets/mug_obstacle_table_move_arm.bddl")]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256(source)
    protocol = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Evaluate observed visual predicates and STL, then global latent forecasts",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "methods": METHODS, "scenarios": SCENARIOS, "waves": waves,
        "fresh_trajectories": len(SCENARIOS) * sum(args.banks_per_wave) * 9,
        "steps_per_trajectory": 96, "frame_rate_hz": 20,
        "policies": ["noop", "low_motion", "safe_arc", "collision_straight", "near_collision",
                     "safe_lateral", "random_smooth", "random_reach", "reach_leave"],
        "calibration": "Historical validation only: balanced accuracy and <=5% validation false-safe operating points; no test-set guarantee",
        "held_out": "Fresh scene seeds, paired across scenarios; no new test outcomes used for tuning",
        "supervision": "State-based scripted data collection; validation physical labels fit scalar thresholds; predicates consume RGB only",
        "formula": "G preserve AND F reach; additionally G preserve AND F G_10frames reach",
        "goal_reference": "Original supplied goal remains fixed under all scene/appearance variations",
        "limits": "Offline candidate ranking is not CEM. Global dynamics only; spatial features are observed, not predicted. Static interventions are not executable trajectories.",
        "source_sha256": hashes,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--banks-per-wave", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--seed", type=int, default=921000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = prepare(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output)}, indent=2))
        return
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               LIBERO_CONFIG_PATH=str(ROOT / ".cache/libero"),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/LIBERO"), env.get("PYTHONPATH", "")]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "4"
    with (output / "supervisor.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_visual_campaign.py"),
                                    "--output", str(output)], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (output / "supervisor.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"status": "launched", "output": str(output), "supervisor_pid": process.pid}, indent=2))


if __name__ == "__main__":
    main()
