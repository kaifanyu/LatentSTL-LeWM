"""Persistent three-GPU queue: calibrate, collect two waves, evaluate, audit forecasts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_stl_lewm.evaluate import sha256


def main(args):
    root = Path(args.output).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    state = {"status": "running", "pid": os.getpid(), "jobs": [], "errors": []}
    lock, report_lock = threading.Lock(), threading.Lock()
    stopping = threading.Event()
    children = {}
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed after freeze: {name}")
    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = root / "status.tmp"; tmp.write_text(json.dumps(state, indent=2) + "\n"); tmp.replace(root / "status.json")
    def stop(sig, frame):
        stopping.set()
        with lock:
            active_children = list(children.values())
        for child in active_children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        state["status"] = "interrupted"; save(); raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def refresh():
        with report_lock:
            completed = subprocess.run([sys.executable, "scripts/report_visual_benchmark.py", str(root)], cwd=ROOT,
                                       capture_output=True, text=True)
            if completed.returncode:
                (root / "report_errors.log").open("a").write(completed.stderr)
    def job(gpu, name, module, arguments):
        if stopping.is_set():
            return False
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [sys.executable, "-m", module, *map(str, arguments)]
        row = {"name": name, "gpu": gpu, "status": "running", "command": command,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
        with lock:
            state["jobs"].append(row); save()
        with (root / "logs" / (name + ".log")).open("x") as log:
            with lock:
                if stopping.is_set():
                    row["status"] = "interrupted"; save()
                    return False
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                children[name] = child; row["pid"] = child.pid; save()
            code = child.wait()
        with lock:
            children.pop(name, None)
            row.update(status="completed" if code == 0 else "failed", exit_code=code,
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
            if code:
                state["errors"].append(name)
            save()
        print(json.dumps({"job": name, "gpu": gpu, "exit_code": code}), flush=True)
        return code == 0
    checkpoint, dataset = protocol["checkpoint"], protocol["dataset"]
    def evaluate(gpu, method, data, output, name, calibration=None):
        a = ["--method", method, "--checkpoint", checkpoint, "--dataset", data, "--output", output]
        if calibration:
            a += ["--calibration", calibration]
        return job(gpu, name, "latent_stl_lewm.visual_predicate_benchmark", a)
    def calibrate(gpu, methods):
        for method in methods:
            if not evaluate(gpu, method, dataset, root / "calibration" / method, "calibrate_" + method):
                raise RuntimeError(f"Calibration failed: {method}")
    save(); refresh()
    try:
        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(calibrate, gpu, methods) for gpu, methods in
                       enumerate((("global",), ("spatial",), ("pixels", "red_pixels")))]
            for future in futures:
                future.result()
        frozen = {method: {"path": str(root / "calibration" / method / "calibration.json"),
                           "sha256": sha256(root / "calibration" / method / "calibration.json")}
                  for method in protocol["methods"]}
        (root / "frozen_calibration.json").write_text(json.dumps(frozen, indent=2) + "\n")
        refresh()
        def worker(gpu):
            for wave in protocol["waves"]:
                for index, scenario in enumerate(protocol["scenarios"]):
                    if stopping.is_set():
                        return
                    if index % 3 != gpu:
                        continue
                    shard = f"{wave['name']}__{scenario}"
                    output = root / "data" / shard
                    ok = job(gpu, "collect_" + shard, "latent_stl_lewm.collect_visual_benchmark",
                             ["--source", dataset, "--output", output, "--scenario", scenario,
                              "--seed", wave["seed"], "--banks", wave["banks"], "--steps", 96])
                    if not ok:
                        continue
                    for method in protocol["methods"]:
                        if stopping.is_set():
                            return
                        if sha256(frozen[method]["path"]) != frozen[method]["sha256"]:
                            raise RuntimeError("Calibration changed during evaluation")
                        evaluate(gpu, method, output / "trajectories.h5", root / "evaluations" / shard / method,
                                 "evaluate_" + shard + "__" + method, frozen[method]["path"])
                    refresh()
            # Each GPU audits a distinct new scene plus the historical diagnostic.
            if stopping.is_set():
                return
            source = dataset if gpu == 0 else root / "data" / (protocol["waves"][0]["name"] + "__" + protocol["scenarios"][gpu]) / "trajectories.h5"
            if Path(source).exists():
                job(gpu, f"forecast_gpu{gpu}", "latent_stl_lewm.visual_forecast_audit",
                    ["--dataset", source, "--checkpoint", checkpoint,
                     "--calibration", frozen["global"]["path"], "--output", root / "forecasts" / f"gpu{gpu}"])
                refresh()
        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(worker, gpu) for gpu in range(3)]
            for future in futures:
                future.result()
        state["status"] = "completed" if not state["errors"] else "completed_with_failures"
    except Exception:
        state["status"] = "failed"
        state["exception"] = traceback.format_exc()
        raise
    finally:
        save(); refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
