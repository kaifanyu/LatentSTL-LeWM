"""Collection scheduling boundaries: explicit EGL routing and valid seed ranges."""
from argparse import Namespace
from dataclasses import asdict
import json

import h5py
import numpy as np
import pytest

from latent_stl_lewm import collect_visual_benchmark as collector
from latent_stl_lewm.simulator import SimulatorConfig


def arguments(tmp_path, **changes):
    values = dict(source=str(tmp_path / "source.h5"), output=str(tmp_path / "collection"),
                  scenario="baseline", seed=921000, banks=4, steps=96, render_gpu=0)
    return Namespace(**(values | changes))


def source_file(path, complete=True, control_freq=20):
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = complete
        handle.attrs["metadata_json"] = json.dumps({
            "simulator_config": asdict(SimulatorConfig(render_gpu=2, control_freq=control_freq)),
        })
        handle.create_dataset("references/goal_rgb", data=np.zeros((128, 128, 3), np.uint8))


@pytest.mark.parametrize("gpu", [0, 1])
def test_worker_routes_egl_independently_of_cuda_mask(tmp_path, monkeypatch, gpu):
    args = arguments(tmp_path, render_gpu=gpu)
    source_file(args.source)
    # A process may see only CUDA device 0 after masking, while EGL must use the
    # physical device selected by the queue. Stop before constructing MuJoCo.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    seen = {}

    class BeforeSimulator(Exception):
        pass

    def intercept(config, scenario):
        seen.update(config=config, scenario=scenario)
        raise BeforeSimulator

    monkeypatch.setattr(collector, "ScenarioEnv", intercept)
    with pytest.raises(BeforeSimulator):
        collector.run(args)
    assert seen["config"]["render_gpu"] == gpu
    assert seen["scenario"] == "baseline"
    assert seen["config"]["goal_tolerance"] == SimulatorConfig().goal_tolerance


@pytest.mark.parametrize("changes, message", [
    ({"banks": 0}, "banks must be positive"),
    ({"banks": -1}, "banks must be positive"),
    ({"steps": 0}, "steps must be at least 2"),
    ({"steps": 1}, "steps must be at least 2"),
    ({"seed": -1}, "unsigned 32-bit"),
    ({"seed": 2**32}, "unsigned 32-bit"),
    ({"seed": 2**32 - 1, "banks": 2}, "unsigned 32-bit"),
    ({"render_gpu": -1}, "nonnegative EGL device"),
])
def test_invalid_schedule_fails_before_creating_output(tmp_path, changes, message):
    args = arguments(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        collector.run(args)
    assert not (tmp_path / "collection").exists()


def test_incomplete_source_is_rejected_before_creating_output(tmp_path):
    args = arguments(tmp_path)
    source_file(args.source, complete=False)
    with pytest.raises(ValueError, match="Source dataset is incomplete"):
        collector.run(args)
    assert not (tmp_path / "collection").exists()


def test_shard_records_actual_control_clock_for_timed_monitoring(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    source_file(args.source, control_freq=10)
    closed = []

    class StopBeforeTrajectories(Exception):
        pass

    class HeaderOnlyEnv:
        action_low, action_high = -np.ones(7), np.ones(7)

        def __init__(self, config, scenario):
            self.config = config

        def metadata(self):
            return {"simulator_config": self.config,
                    "control_freq_hz": self.config["control_freq"],
                    "frame_freq_hz": self.config["control_freq"], "action_repeat": 1}

        def reset(self, seed):
            raise StopBeforeTrajectories

        def close(self):
            closed.append(True)

    monkeypatch.setattr(collector, "ScenarioEnv", HeaderOnlyEnv)
    with pytest.raises(StopBeforeTrajectories):
        collector.run(args)
    assert closed == [True]
    assert not (tmp_path / "collection/trajectories.h5").exists()
    with h5py.File(tmp_path / "collection/trajectories.partial.h5", "r") as handle:
        assert not handle.attrs["complete"]
        assert handle.attrs["control_freq_hz"] == 10
        assert handle.attrs["frame_freq_hz"] == 10
        assert handle.attrs["action_repeat"] == 1
        nested = json.loads(handle.attrs["metadata_json"])
        assert nested["control_freq_hz"] == handle.attrs["control_freq_hz"]
