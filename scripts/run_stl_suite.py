"""Prepare and supervise a finite, reproducible two-GPU visual-STL experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["global", "spatial", "pixels", "red_pixels"]
SCENARIOS = ["baseline", "start_jitter_3cm", "start_x_plus3cm", "mug_x_plus3cm",
             "mug_x_minus3cm", "mug_y_plus3cm", "camera_x_plus2cm", "dim_085"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def make_protocol(runtime_root, smoke=False):
    r = Path(runtime_root)
    waves = ([{"name": "smoke", "seed": 941000, "banks": 1, "steps": 96}] if smoke else
             [{"name": "wave1", "seed": 951000, "banks": 4, "steps": 96},
              {"name": "wave2", "seed": 952000, "banks": 8, "steps": 96},
              {"name": "wave3_long", "seed": 953000, "banks": 12, "steps": 192}])
    scenarios = ["baseline", "mug_x_plus3cm"] if smoke else SCENARIOS
    jobs = []

    def add(name, module, args, needs=(), priority=5, gpu=None):
        jobs.append({"name": name, "module": module, "arguments": list(map(str, args)),
                     "needs": list(needs), "priority": priority, "preferred_gpu": gpu,
                     "log": f"logs/{name}.log"})

    counts = [16, 8, 8] if smoke else [512, 64, 64]
    dataset = r / "dataset/trajectories.h5"
    reference = r / "reference_dataset/trajectories.h5"
    for name, output, split_counts, gpu in (
            ("collect_source", r / "dataset", counts, 0),
            ("collect_reference", r / "reference_dataset", [8, 8, 8], 1)):
        add(name, "latent_stl_lewm.collect", ["--output", output, "--train-episodes", split_counts[0],
            "--val-episodes", split_counts[1], "--test-episodes", split_counts[2], "--steps", 96,
            "--seed", 20261004, "--render-gpu", "{gpu}"], priority=0, gpu=gpu)
    add("audit_source", "latent_stl_lewm.dataset", [dataset, "--output", r / "dataset_audit.json",
        "--videos-dir", r / "dataset_videos"], ["collect_source"], priority=0)
    add("verify_references", "scripts.run_stl_suite", ["--output", r, "--verify-references"],
        ["collect_source", "collect_reference"], priority=0)
    for wave in waves:
        for scenario in scenarios:
            shard = f"{wave['name']}__{scenario}"
            add("collect_" + shard, "latent_stl_lewm.collect_visual_benchmark",
                ["--source", reference, "--output", r / "data" / shard, "--scenario", scenario,
                 "--seed", wave["seed"], "--banks", wave["banks"], "--steps", wave["steps"],
                 "--render-gpu", "{gpu}"], ["collect_reference"], priority=5)
    for seed in (0, 1):
        prefix = f"seed_{seed}"
        model = r / "models" / prefix
        train, fine = f"train_{prefix}", f"finetune_{prefix}"
        add(train, "latent_stl_lewm.train", ["--dataset", dataset, "--output", model / "train",
            "--device", "cuda", "--seed", seed, "--batch-size", 32, "--amp", "--max-steps",
            2 if smoke else 20000, "--epochs", 100, "--val-every", 2 if smoke else 500,
            "--checkpoint-every", 2 if smoke else 500], ["audit_source", "verify_references"],
            priority=1, gpu=seed)
        add(fine, "latent_stl_lewm.finetune_multistep", ["--checkpoint", model / "train/best.pt",
            "--dataset", dataset, "--output", model / "finetune", "--max-steps", 2 if smoke else 3000,
            "--future-steps", 8, "--batch-size", 32, "--encode-batch-size", 32, "--amp", "--seed", seed,
            "--validation-windows", 32 if smoke else 0, "--val-every", 2 if smoke else 250],
            [train], priority=2, gpu=seed)
        checkpoint = model / "finetune/best.pt"
        calibration_jobs = []
        for method in METHODS:
            name = f"calibrate_{prefix}_{method}"
            calibration_jobs.append(name)
            add(name, "latent_stl_lewm.visual_predicate_benchmark", ["--method", method,
                "--dataset", dataset, "--checkpoint", checkpoint, "--output", model / "calibration" / method],
                [fine], priority=3)
        freeze = f"freeze_{prefix}"
        add(freeze, "scripts.run_stl_suite", ["--output", r, "--freeze-seed", seed],
            calibration_jobs, priority=3)
        for wave in waves:
            for scenario in scenarios:
                shard = f"{wave['name']}__{scenario}"
                fresh = r / "data" / shard / "trajectories.h5"
                for method in METHODS:
                    add(f"evaluate_{prefix}_{shard}_{method}", "latent_stl_lewm.visual_predicate_benchmark",
                        ["--method", method, "--dataset", fresh, "--checkpoint", checkpoint,
                         "--calibration", model / "calibration" / method / "calibration.json",
                         "--output", model / "evaluations" / shard / method],
                        [freeze, "collect_" + shard], priority=4)
                if wave is waves[0] or wave["steps"] == 192:
                    add(f"forecast_{prefix}_{shard}", "latent_stl_lewm.visual_forecast_audit",
                        ["--dataset", fresh, "--checkpoint", checkpoint, "--calibration",
                         model / "calibration/global/calibration.json", "--output", model / "forecasts" / shard],
                        [freeze, "collect_" + shard], priority=6)
    return {"schema_version": 1, "created_at_utc": now(), "runtime_root": str(r), "smoke": smoke,
            "gpus": [0, 1], "training_seeds": [0, 1], "methods": METHODS, "scenarios": scenarios,
            "waves": waves, "source_split_counts": dict(zip(("train", "val", "test"), counts)),
            "source_seed": 20261004, "frame_freq_hz": 20,
            "fresh_trajectories": len(scenarios) * sum(w["banks"] for w in waves) * 9,
            "physical_tolerances": {"displacement_m": .002, "rotation_deg": 2, "goal_m": .03},
            "training_steps_per_seed": 2 if smoke else 20000,
            "finetune_steps_per_seed": 2 if smoke else 3000,
            "formulas": ["G preserve", "F reach", "G preserve AND F reach",
                         "G preserve AND F (10 consecutive observations satisfy reach)"],
            "operating_points": ["balanced", "false_safe_05"],
            "calibration": "Source validation labels only; thresholds frozen before fresh evaluation",
            "scope": "All four implemented visual preserve/reach representations; offline monitoring and ranking",
            "jobs": jobs}


def prepare(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "data", "models", "source_snapshot"):
        (root / name).mkdir()
    snapshot = root / "source_snapshot"
    hashes = {}
    for folder in ("latent_stl_lewm", "scripts", "tests", "docs"):
        for source in sorted((ROOT / folder).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[str(relative)] = digest(source)
    for name in ("pyproject.toml", "requirements.lock.txt", "Dockerfile", "compose.yaml", "README.md"):
        shutil.copy2(ROOT / name, snapshot / name)
        hashes[name] = digest(ROOT / name)
    for name in ("third_party", ".venv", ".cache"):
        (snapshot / name).symlink_to(Path("/workspace") / name, target_is_directory=True)
    protocol = make_protocol(args.runtime_root or str(root), args.smoke)
    protocol["source_sha256"] = hashes
    atomic(root / "protocol.json", protocol)
    if (ROOT / "docs/stl_suite.md").exists():
        shutil.copy2(ROOT / "docs/stl_suite.md", root / "TRIALS.md")
    print(json.dumps({"prepared": str(root), "jobs": len(protocol["jobs"]),
                      "fresh_trajectories": protocol["fresh_trajectories"]}, indent=2))


def verify_references(root):
    import h5py
    import numpy as np
    p = json.loads((root / "protocol.json").read_text())
    with h5py.File(root / "dataset/trajectories.h5") as source, h5py.File(root / "reference_dataset/trajectories.h5") as ref:
        for name in ("goal_rgb", "safe_rgb"):
            if not np.array_equal(source[f"references/{name}"][:], ref[f"references/{name}"][:]):
                raise ValueError(f"Parallel reference bootstrap differs: {name}")
        seeds = {int(ep.attrs["scene_seed"]) for ep in source["trajectories"].values()}
        fresh = {w["seed"] + i for w in p["waves"] for i in range(w["banks"])}
        if seeds & fresh:
            raise ValueError("Source and fresh evaluation scene seeds overlap")
    atomic(root / "reference_audit.json", {"equal_reference_images": True,
        "fresh_seeds_disjoint_from_source": True,
        "note": "24 reference-bootstrap episodes duplicate source starts; excluded from evaluation/training counts"})


def freeze(root, seed):
    model = root / "models" / f"seed_{seed}"
    atomic(model / "frozen_calibration.json", {m: {"path": str(model / "calibration" / m / "calibration.json"),
        "sha256": digest(model / "calibration" / m / "calibration.json")} for m in METHODS})


def ready_jobs(jobs, gpu):
    states = {j["name"]: j["status"] for j in jobs}
    return sorted([j for j in jobs if j["status"] == "queued"
        and j.get("preferred_gpu") in (None, gpu)
        and all(states[n] == "completed" for n in j["needs"])], key=lambda j: j["priority"])


def validate_protocol(protocol):
    jobs = protocol["jobs"]
    gpus = protocol["gpus"]
    if not gpus or any(type(g) is not int or g < 0 for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers with at least one device")
    names = {j["name"] for j in jobs}
    if len(names) != len(jobs):
        raise ValueError("Duplicate job names")
    reached = set()
    for job in jobs:
        if not set(job["needs"]) <= names:
            raise ValueError("Unknown job dependency")
        if job.get("preferred_gpu") not in (None, *protocol["gpus"]):
            raise ValueError("Job requests an unavailable GPU")
    while len(reached) < len(jobs):
        available = {j["name"] for j in jobs if set(j["needs"]) <= reached} - reached
        if not available:
            raise ValueError("Job dependencies contain a cycle")
        reached.update(available)


def supervise(args):
    import fcntl
    root = Path(args.output).resolve()
    lock = (root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "status.json").exists():
        raise ValueError("This run already has execution state; use a new run directory")
    protocol = json.loads((root / "protocol.json").read_text())
    validate_protocol(protocol)
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    state = {"status": "running", "pid": os.getpid(), "started_at_utc": now(), "jobs":
             [dict(j, status="queued") for j in protocol["jobs"]]}
    active, stopped, stop_deadline = {}, False, None
    (root / "supervisor.pid").write_text(f"{os.getpid()}\n")

    def save():
        state["updated_at_utc"] = now()
        atomic(root / "status.json", state)

    def stop(signum, frame):
        nonlocal stopped, stop_deadline
        stopped = True
        stop_deadline = time.monotonic() + 30
        state["status"] = "stopping"
        save()
        for proc, _, _ in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env_base = os.environ.copy()
    env_base.update(PYTHONPATH=os.pathsep.join([str(ROOT), "/workspace/third_party/LIBERO"]),
                    PYTHONUNBUFFERED="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env_base[name] = "4"
    report_proc, report_log, next_report = None, None, 0

    def refresh(final=False):
        nonlocal report_proc, report_log, next_report
        if report_proc is not None:
            if report_proc.poll() is None:
                return
            report_log.close()
        report_log = (root / "report.log").open("a")
        report_proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/report_stl_suite.py"), str(root)],
            cwd=ROOT, env=env_base, stdout=report_log, stderr=subprocess.STDOUT)
        next_report = time.monotonic() + 60
        if final:
            report_proc.wait()
            report_log.close()

    try:
        save()
        while True:
            if stopped and time.monotonic() >= stop_deadline:
                for proc, _, _ in active.values():
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            for gpu, (proc, row, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                row.update(status="interrupted" if stopped else ("completed" if code == 0 else "failed"),
                           exit_code=code, finished_at_utc=now())
                del active[gpu]
                print(json.dumps({"job": row["name"], "gpu": gpu, "exit_code": code}), flush=True)
                save()
            states = {j["name"]: j["status"] for j in state["jobs"]}
            for row in state["jobs"]:
                if row["status"] == "queued" and any(states[n] in ("failed", "blocked", "interrupted") for n in row["needs"]):
                    row.update(status="blocked", reason="Required upstream job did not complete")
            if not stopped:
                for gpu in protocol["gpus"]:
                    if gpu in active:
                        continue
                    ready = ready_jobs(state["jobs"], gpu)
                    if not ready:
                        continue
                    row = ready[0]
                    # Verify the frozen calibration artifact before every dependent evaluation.
                    if "--calibration" in row["arguments"]:
                        path = Path(row["arguments"][row["arguments"].index("--calibration") + 1])
                        frozen = json.loads((path.parents[2] / "frozen_calibration.json").read_text())
                        if digest(path) != frozen[path.parent.name]["sha256"]:
                            raise ValueError("Calibration changed after freeze")
                    command = [sys.executable, "-m", row["module"], *[a.replace("{gpu}", str(gpu)) for a in row["arguments"]]]
                    env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
                    log = (root / row["log"]).open("x")
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
                    row.update(status="running", command=command, gpu=gpu, pid=proc.pid,
                               started_at_utc=now(), environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")})
                    active[gpu] = (proc, row, log)
                    save()
            if time.monotonic() >= next_report:
                save(); refresh()
            if not active and (stopped or all(j["status"] != "queued" for j in state["jobs"])):
                break
            if not active and not any(ready_jobs(state["jobs"], gpu) for gpu in protocol["gpus"]):
                # Propagate upstream failures through the entire dependency graph.
                if any(j["status"] == "queued" for j in state["jobs"]):
                    continue
            time.sleep(2)
        state["status"] = "interrupted" if stopped else ("completed" if all(j["status"] == "completed" for j in state["jobs"]) else "completed_with_failures")
        if stopped:
            for row in state["jobs"]:
                if row["status"] == "queued":
                    row.update(status="cancelled", reason="Supervisor was stopped")
    except BaseException as exc:
        stop(signal.SIGTERM, None)
        state.update(status="failed", error=repr(exc))
        for proc, row, log in active.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            row.update(status="interrupted", exit_code=proc.returncode, finished_at_utc=now())
            log.close()
        raise
    finally:
        state["finished_at_utc"] = now()
        save()
        if report_proc is not None and report_proc.poll() is None:
            report_proc.wait()
        refresh(final=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--runtime-root")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verify-references", action="store_true")
    p.add_argument("--freeze-seed", type=int)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.verify_references:
        verify_references(Path(args.output))
    elif args.freeze_seed is not None:
        freeze(Path(args.output), args.freeze_seed)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
