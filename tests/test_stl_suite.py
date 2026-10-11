"""Protocol and scheduler checks independent of CUDA or simulator performance."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "stl_suite_under_test", Path(__file__).resolve().parents[1] / "scripts/run_stl_suite.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def option(job, name):
    return job["arguments"][job["arguments"].index(name) + 1]


def ancestors(jobs, name):
    pending, found = list(jobs[name]["needs"]), set()
    while pending:
        dependency = pending.pop()
        if dependency not in found:
            found.add(dependency)
            pending.extend(jobs[dependency]["needs"])
    return found


@pytest.mark.parametrize("smoke, expected_rollouts", [(False, 1728), (True, 18)])
def test_protocol_has_complete_acyclic_dependencies_and_paired_data(smoke, expected_rollouts):
    protocol = suite.make_protocol("/workspace/runs/test_suite", smoke)
    suite.validate_protocol(protocol)
    jobs = {job["name"]: job for job in protocol["jobs"]}
    assert len(jobs) == len(protocol["jobs"])
    for name, job in jobs.items():
        assert set(job["needs"]) <= jobs.keys()
        assert name not in ancestors(jobs, name)
        assert job["preferred_gpu"] in [None, *protocol["gpus"]]
    assert protocol["fresh_trajectories"] == expected_rollouts
    assert jobs["collect_source"]["preferred_gpu"] == 0
    assert jobs["collect_reference"]["preferred_gpu"] == 1
    for seed in protocol["training_seeds"]:
        prefix = f"seed_{seed}"
        assert jobs[f"train_{prefix}"]["preferred_gpu"] == seed
        assert {"audit_source", "verify_references"} <= set(jobs[f"train_{prefix}"]["needs"])
        assert jobs[f"finetune_{prefix}"]["needs"] == [f"train_{prefix}"]
        assert set(jobs[f"freeze_{prefix}"]["needs"]) == {
            f"calibrate_{prefix}_{method}" for method in protocol["methods"]}
        for method in protocol["methods"]:
            assert jobs[f"calibrate_{prefix}_{method}"]["needs"] == [f"finetune_{prefix}"]
        for wave in protocol["waves"]:
            for scenario in protocol["scenarios"]:
                shard = f"{wave['name']}__{scenario}"
                collector = jobs["collect_" + shard]
                assert collector["needs"] == ["collect_reference"]
                assert option(collector, "--render-gpu") == "{gpu}"
                assert int(option(collector, "--steps")) == wave["steps"]
                assert int(option(collector, "--banks")) == wave["banks"]
                for method in protocol["methods"]:
                    evaluator = jobs[f"evaluate_{prefix}_{shard}_{method}"]
                    assert set(evaluator["needs"]) == {f"freeze_{prefix}", "collect_" + shard}
                    assert option(evaluator, "--dataset") == option(collector, "--output") + "/trajectories.h5"
                    assert option(evaluator, "--method") == method
                    assert "verify_references" in ancestors(jobs, evaluator["name"])
                if wave is protocol["waves"][0] or wave["steps"] == 192:
                    forecaster = jobs[f"forecast_{prefix}_{shard}"]
                    assert set(forecaster["needs"]) == {f"freeze_{prefix}", "collect_" + shard}


def test_full_matrix_balances_durations_without_source_or_wave_seed_leakage():
    p = suite.make_protocol("/workspace/runs/test_suite")
    counts = {}
    fresh_seeds = set()
    for wave in p["waves"]:
        seeds = set(range(wave["seed"], wave["seed"] + wave["banks"]))
        assert not fresh_seeds & seeds
        fresh_seeds |= seeds
        counts[wave["steps"]] = counts.get(wave["steps"], 0) + len(p["scenarios"]) * wave["banks"] * 9
    assert counts == {96: 864, 192: 864}
    assert sum(steps * number for steps, number in counts.items()) == 248832
    source_seeds = set()
    for split_index, count in enumerate(p["source_split_counts"].values()):
        source_seeds |= {p["source_seed"] + (split_index + 1) * 100000 + bank
                         for bank in range((count + 7) // 8)}
    assert not source_seeds & fresh_seeds
    assert len(fresh_seeds) == 24
    assert [wave["seed"] for wave in p["waves"]] == [951000, 952000, 953000]
    smoke = suite.make_protocol("/workspace/runs/preflight", smoke=True)
    smoke_seeds = {wave["seed"] + bank for wave in smoke["waves"] for bank in range(wave["banks"])}
    assert not fresh_seeds & smoke_seeds


@pytest.mark.parametrize("jobs, message", [
    ([dict(name="a", needs=[]), dict(name="a", needs=[])], "Duplicate job names"),
    ([dict(name="a", needs=["missing"])], "Unknown job dependency"),
    ([dict(name="a", needs=["b"]), dict(name="b", needs=["a"])], "contain a cycle"),
    ([dict(name="a", needs=["a"])], "contain a cycle"),
    ([dict(name="a", needs=[], preferred_gpu=2)], "unavailable GPU"),
])
def test_invalid_dag_is_rejected_before_scheduling(jobs, message):
    with pytest.raises(ValueError, match=message):
        suite.validate_protocol(dict(gpus=[0, 1], jobs=jobs))


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1], ["0"]])
def test_invalid_gpu_pool_is_rejected(gpus):
    with pytest.raises(ValueError, match="GPU"):
        suite.validate_protocol(dict(gpus=gpus, jobs=[dict(name="a", needs=[])]))


def test_ready_jobs_respects_gpu_dependencies_and_failed_upstream():
    jobs = [
        dict(name="complete", status="completed", needs=[], priority=0),
        dict(name="failed", status="failed", needs=[], priority=0),
        dict(name="low", status="queued", needs=["complete"], priority=5, preferred_gpu=None),
        dict(name="high", status="queued", needs=["complete"], priority=1, preferred_gpu=1),
        dict(name="dependent", status="queued", needs=["failed"], priority=0, preferred_gpu=None),
        dict(name="running", status="running", needs=[], priority=0, preferred_gpu=None),
    ]
    assert [j["name"] for j in suite.ready_jobs(jobs, 0)] == ["low"]
    assert [j["name"] for j in suite.ready_jobs(jobs, 1)] == ["high", "low"]


def reference_fixture(root, source_seed, different_goal=False):
    for directory in ("dataset", "reference_dataset"):
        (root / directory).mkdir()
        with h5py.File(root / directory / "trajectories.h5", "w") as handle:
            for name in ("goal_rgb", "safe_rgb"):
                value = int(different_goal and directory == "reference_dataset" and name == "goal_rgb")
                handle.create_dataset("references/" + name, data=np.full((2, 2, 3), value, np.uint8))
            handle.create_group("trajectories/0").attrs["scene_seed"] = source_seed
    (root / "protocol.json").write_text(json.dumps({"waves": [dict(seed=941000, banks=4)]}))


def test_reference_audit_checks_cross_gpu_images_and_fresh_seed_provenance(tmp_path):
    reference_fixture(tmp_path, source_seed=20361004)
    suite.verify_references(tmp_path)
    report = json.loads((tmp_path / "reference_audit.json").read_text())
    assert report["equal_reference_images"]
    assert report["fresh_seeds_disjoint_from_source"]


@pytest.mark.parametrize("seed, changed_goal, message", [
    (20361004, True, "reference bootstrap differs"),
    (941002, False, "scene seeds overlap"),
])
def test_reference_audit_fails_closed(tmp_path, seed, changed_goal, message):
    reference_fixture(tmp_path, seed, changed_goal)
    with pytest.raises(ValueError, match=message):
        suite.verify_references(tmp_path)
    assert not (tmp_path / "reference_audit.json").exists()


def test_source_snapshot_change_prevents_any_execution(tmp_path, monkeypatch):
    root, code = tmp_path / "run", tmp_path / "code"
    root.mkdir(); code.mkdir()
    source = code / "worker.py"
    source.write_text("print('frozen')\n")
    protocol = dict(source_sha256={"worker.py": suite.digest(source)}, gpus=[0, 1], jobs=[])
    (root / "protocol.json").write_text(json.dumps(protocol))
    source.write_text("print('changed')\n")
    monkeypatch.setattr(suite, "ROOT", code)
    with pytest.raises(ValueError, match="Frozen source changed"):
        suite.supervise(Namespace(output=str(root)))
    assert not (root / "status.json").exists()
    assert not (root / "supervisor.pid").exists()


def test_changed_frozen_calibration_is_not_used_for_evaluation(tmp_path, monkeypatch):
    root, code = tmp_path / "run", tmp_path / "code"
    (root / "logs").mkdir(parents=True)
    (code / "scripts").mkdir(parents=True)
    (code / "scripts/report_stl_suite.py").write_text("pass\n")
    for method in suite.METHODS:
        directory = root / "models/seed_0/calibration" / method
        directory.mkdir(parents=True)
        (directory / "calibration.json").write_text("{}\n")
    suite.freeze(root, 0)
    calibration = root / "models/seed_0/calibration/global/calibration.json"
    calibration.write_text('{"threshold": "changed"}\n')
    protocol = dict(source_sha256={}, gpus=[0, 1], jobs=[dict(
        name="evaluate", module="worker_must_not_run", arguments=["--calibration", str(calibration)],
        needs=[], priority=0, preferred_gpu=None, log="logs/evaluate.log")])
    (root / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(suite, "ROOT", code)
    monkeypatch.setattr(suite.signal, "signal", lambda *args: None)
    with pytest.raises(ValueError, match="Calibration changed after freeze"):
        suite.supervise(Namespace(output=str(root)))
    state = json.loads((root / "status.json").read_text())
    assert state["status"] == "failed"
    assert not (root / "logs/evaluate.log").exists()


def test_supervisor_runs_on_both_gpus_and_blocks_failed_descendants(tmp_path, monkeypatch):
    """Use real child processes to check environment routing and failure cleanup."""
    root, code = tmp_path / "run", tmp_path / "code"
    (root / "logs").mkdir(parents=True)
    (code / "scripts").mkdir(parents=True)
    (code / "scripts/report_stl_suite.py").write_text("pass\n")
    (code / "worker.py").write_text(
        "import json, os, pathlib, sys, time\n"
        "time.sleep(0.02)\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({k: os.environ[k] "
        "for k in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_EGL_DEVICE_ID')}))\n"
        "sys.exit(int(sys.argv[2]))\n")

    def job(name, gpu, dependencies=(), exit_code=0):
        return dict(name=name, module="worker", arguments=[str(root / f"{name}.json"), str(exit_code)],
                    needs=list(dependencies), priority=0, preferred_gpu=gpu, log=f"logs/{name}.log")

    protocol = dict(source_sha256={}, gpus=[0, 1], jobs=[
        job("source0", 0), job("source1", 1, exit_code=3),
        job("train0", 0, ["source0"]), job("eval", None, ["train0", "source1"]),
        job("aggregate", None, ["eval"]),
    ])
    (root / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(suite, "ROOT", code)
    monkeypatch.setattr(suite.signal, "signal", lambda *args: None)
    monkeypatch.setattr(suite.time, "sleep", lambda seconds: None)
    suite.supervise(Namespace(output=str(root)))
    state = json.loads((root / "status.json").read_text())
    assert state["status"] == "completed_with_failures"
    statuses = {row["name"]: row["status"] for row in state["jobs"]}
    assert statuses == dict(source0="completed", source1="failed", train0="completed",
                            eval="blocked", aggregate="blocked")
    for gpu in (0, 1):
        env = json.loads((root / f"source{gpu}.json").read_text())
        assert env == {"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu)}
    assert not (root / "eval.json").exists()
    assert not (root / "aggregate.json").exists()


def test_stop_kills_stubborn_child_after_deadline_and_cancels_queued_work(tmp_path, monkeypatch):
    """A real child ignores SIGTERM; accelerate only the supervisor's clock."""
    root, code = tmp_path / "run", tmp_path / "code"
    (root / "logs").mkdir(parents=True)
    (code / "scripts").mkdir(parents=True)
    (code / "scripts/report_stl_suite.py").write_text("pass\n")
    marker = root / "child_ready"
    (code / "worker.py").write_text(
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready')\n"
        "time.sleep(60)\n")

    def job(name, dependencies=(), priority=1):
        return dict(name=name, module="worker", arguments=[str(marker)], needs=list(dependencies),
                    priority=priority, preferred_gpu=0, log=f"logs/{name}.log")

    protocol = dict(source_sha256={}, gpus=[0], jobs=[
        job("stubborn", priority=0), job("queued"), job("dependent", ["stubborn"]),
    ])
    (root / "protocol.json").write_text(json.dumps(protocol))
    handlers, sent_signals = {}, []
    real_sleep, real_clock, real_killpg = time.sleep, time.monotonic, suite.os.killpg
    started = real_clock()
    advance, stop_sent = 0, False

    def fast_sleep(seconds):
        nonlocal advance, stop_sent
        if marker.exists() and not stop_sent:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            stop_sent = True
            advance = 31
        if real_clock() - started > 5:
            raise TimeoutError("Supervisor failed to stop its test child")
        real_sleep(.005)

    def killpg(pid, signum):
        sent_signals.append(signum)
        return real_killpg(pid, signum)

    monkeypatch.setattr(suite, "ROOT", code)
    monkeypatch.setattr(suite, "time", SimpleNamespace(monotonic=lambda: real_clock() + advance, sleep=fast_sleep))
    monkeypatch.setattr(suite.signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(suite.os, "killpg", killpg)
    suite.supervise(Namespace(output=str(root)))
    state = json.loads((root / "status.json").read_text())
    rows = {row["name"]: row for row in state["jobs"]}
    assert state["status"] == "interrupted"
    assert rows["stubborn"]["status"] == "interrupted"
    assert rows["stubborn"]["exit_code"] == -signal.SIGKILL
    assert rows["queued"]["status"] == "cancelled"
    assert rows["dependent"]["status"] == "blocked"
    assert signal.SIGTERM in sent_signals and signal.SIGKILL in sent_signals
    assert not (root / "logs/queued.log").exists()
    assert not (root / "logs/dependent.log").exists()
