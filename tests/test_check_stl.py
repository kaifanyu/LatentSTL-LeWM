import argparse
import json

import h5py
import numpy as np
import pytest

from latent_stl_lewm.check_stl import default_formula, episode_timestamps, physical_predicates, run
from latent_stl_lewm.timed_stl import formula_horizon


def make_dataset(path, *, frequency=20):
    with h5py.File(path, "w") as data:
        data.attrs["complete"] = True
        data.attrs["control_freq_hz"] = frequency
        data.create_group("references").create_dataset("goal_rgb", data=np.zeros((128, 128, 3), np.uint8))
        episodes = data.create_group("trajectories")
        for key, split, codes in (("validation", "val", [0, 20, 80, 100]),
                                  ("test", "test", np.zeros(97, dtype=np.uint8))):
            ep = episodes.create_group(key)
            ep.attrs["split"] = split
            images = np.zeros((len(codes), 128, 128, 3), np.uint8)
            images[:, 0, 0, 0] = codes
            ep.create_dataset("rgb", data=images)
            ep.create_dataset("control_step", data=np.arange(len(codes)))
            diagnostics = ep.create_group("diagnostics")
            diagnostics.create_dataset("safe", data=np.asarray(codes) <= 20)
            diagnostics.create_dataset("reach", data=np.asarray(codes) >= 80)


class FakeFeatures:
    calls = []
    artifact_required_for_test = None

    def __init__(self, method, checkpoint, device, batch_size):
        self.method = method

    def traces(self, rgb, goal):
        self.calls.append(rgb.parent.attrs["split"])
        if rgb.parent.attrs["split"] == "test" and self.artifact_required_for_test:
            assert self.artifact_required_for_test.exists(), "Test features accessed before calibration was frozen"
        distance = np.asarray(rgb[:, 0, 0, 0], dtype=float) / 100
        return distance, 1 - distance, (40, 40, 70, 70), False


def options(tmp_path, dataset, **updates):
    args = dict(dataset=dataset, checkpoint=None, methods=["pixels", "red_pixels"],
                output=tmp_path / "result", formula=None, deadline=5., hold=0.,
                max_episodes=1, max_calibration_episodes=None, calibration_artifact=None,
                operating_point="balanced", device="cpu", batch_size=4)
    args.update(updates)
    return argparse.Namespace(**args)


def test_validation_freeze_timestamps_pending_and_artifact_reuse(tmp_path):
    dataset = tmp_path / "data.h5"
    make_dataset(dataset)
    args = options(tmp_path, dataset)
    FakeFeatures.calls = []
    FakeFeatures.artifact_required_for_test = args.output / "calibration.json"
    report = run(args, extractor_factory=FakeFeatures)
    assert FakeFeatures.calls == ["val", "val", "test", "test"]
    assert report["episodes"][0]["duration_seconds"] == 4.8
    assert report["episodes"][0]["physical"]["status"] == "pending"
    assert report["methods"]["pixels"]["visual_status_counts"]["pending"] == 1
    assert report["methods"]["pixels"]["decided_agreement"] is None
    traces = json.loads((args.output / "traces/episode_00000.json").read_text())
    assert traces["timestamps_seconds"][-1] == 4.8
    assert traces["physical"]["upper"] == 1.0
    artifact = json.loads((args.output / "calibration.json").read_text())
    assert artifact["validation_episode_ids"] == ["validation"]
    FakeFeatures.calls = []
    reused = options(tmp_path, dataset, output=tmp_path / "reused", calibration_artifact=args.output / "calibration.json")
    FakeFeatures.artifact_required_for_test = reused.output / "calibration.json"
    assert run(reused, extractor_factory=FakeFeatures)["methods"] == report["methods"]
    assert FakeFeatures.calls == ["test", "test"]
    with pytest.raises(FileExistsError):
        run(args, extractor_factory=FakeFeatures)


def test_declared_physical_margins_preserve_translation_and_rotation(tmp_path):
    path = tmp_path / "physical.h5"
    make_dataset(path)
    with h5py.File(path, "a") as data:
        ep = data["trajectories/validation"]
        d = ep["diagnostics"]
        d["goal_error"] = [.06, .03, .02, 0]
        d["object_displacement"] = [0, .001, .003, 0]
        d["object_rotation_deg"] = [0, 0, 0, 4]
        predicates, kind = physical_predicates(ep, {"goal_tolerance": .03, "displacement_tolerance": .002,
                                                    "rotation_tolerance_deg": 2})
    np.testing.assert_allclose(predicates["reach"], [-1, 0, 1/3, 1])
    np.testing.assert_allclose(predicates["preserve"], [1, .5, -.5, -1])
    assert "physical margins" in kind


@pytest.mark.parametrize("frequency", [0, float("nan"), -20])
def test_invalid_frequency_rejected_before_output(tmp_path, frequency):
    path = tmp_path / "invalid.h5"
    make_dataset(path, frequency=frequency)
    args = options(tmp_path, path)
    with pytest.raises(ValueError, match="control_freq_hz"):
        run(args, extractor_factory=FakeFeatures)
    assert not args.output.exists()


def test_missing_or_duplicate_control_timestamps_are_not_invented(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    with h5py.File(path, "a") as data:
        ep = data["trajectories/validation"]
        ep["control_step"][1] = 0
        with pytest.raises(ValueError, match="increasing"):
            episode_timestamps(ep, 20)
        del ep["control_step"]
        with pytest.raises(ValueError, match="missing"):
            episode_timestamps(ep, 20)


def test_custom_formula_no_fabricated_ground_truth(tmp_path):
    path = tmp_path / "formula.json"
    path.write_text(json.dumps({"op": "predicate", "name": "grasped"}))
    args = options(tmp_path, tmp_path / "unused.h5", formula=path)
    with pytest.raises(ValueError, match="no physical labels are inferred"):
        run(args, extractor_factory=FakeFeatures)


def test_hold_must_finish_before_deadline():
    formula = default_formula(5, 1)
    assert formula["args"][1]["interval"] == [0, 4]
    assert formula_horizon(formula) == 5
    with pytest.raises(ValueError):
        default_formula(5, 6)


def test_reused_artifact_rejects_dataset_mismatch(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps({"schema_version": 1, "calibration_split": "val", "dataset_sha256": "wrong"}))
    with pytest.raises(ValueError, match="dataset hash"):
        run(options(tmp_path, path, calibration_artifact=artifact), extractor_factory=FakeFeatures)
