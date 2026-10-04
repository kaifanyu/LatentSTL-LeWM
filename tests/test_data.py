"""Data boundary tests; optional real simulator replay test uses GPU2."""
import os

import h5py
import numpy as np
import pytest
import torch

from latent_stl_lewm.dataset import TrajectoryWindowDataset, training_action_statistics


@pytest.fixture
def dataset_file(tmp_path):
    path = tmp_path / "unit_fixture.h5"
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = True
        trajectories = f.create_group("trajectories")
        for i, split in enumerate(("train", "val", "test")):
            ep = trajectories.create_group(str(i))
            ep.attrs["split"] = split
            ep.attrs["bank_id"] = split + "_0000"
            ep.create_dataset("rgb", data=np.full((6, 8, 8, 3), i * 40, dtype=np.uint8))
            ep.create_dataset("action", data=np.arange(5)[:, None] * np.ones((1, 7)) + i * 100)
            ep.create_dataset("diagnostics/secret_goal_xyz", data=np.ones((6, 3)))
    return path


def test_windows_preserve_alignment_and_split(dataset_file):
    dataset = TrajectoryWindowDataset(dataset_file, "train", context=3, horizon=1)
    assert len(dataset) == 3
    assert dataset.trajectory_ids == ["0"]
    last = dataset[2]
    assert set(last) == {"rgb", "action"}
    assert last["rgb"].shape == (4, 3, 8, 8)
    assert last["rgb"].dtype == torch.float32
    assert last["action"].shape == (3, 7)
    assert last["action"][:, 0].tolist() == [2, 3, 4]
    assert dataset[0]["rgb"].max() == 0
    dataset.close()


def test_train_statistics_exclude_other_splits(dataset_file):
    mean, std = training_action_statistics(dataset_file)
    np.testing.assert_allclose(mean, np.full(7, 2))
    np.testing.assert_allclose(std, np.full(7, np.std(np.arange(5))))


def test_misaligned_episode_rejected(dataset_file):
    with h5py.File(dataset_file, "a") as f:
        del f["trajectories/0/action"]
        f.create_dataset("trajectories/0/action", data=np.zeros((2, 7)))
    with pytest.raises(ValueError, match="Misaligned"):
        TrajectoryWindowDataset(dataset_file)


def test_incomplete_collection_rejected(dataset_file):
    with h5py.File(dataset_file, "a") as f:
        f.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        TrajectoryWindowDataset(dataset_file)


@pytest.mark.skipif(os.environ.get("RUN_SIMULATOR_TESTS") != "1", reason="opt-in real EGL/MuJoCo integration")
def test_real_simulator_exact_replay():
    from latent_stl_lewm.simulator import MugPreservationEnv
    env = MugPreservationEnv()
    try:
        env.reset(0)
        snapshot = env.snapshot()
        action = env.action_to_target(env.config.goal_eef)
        states, starts, frames = [], [], []
        for _ in range(3):
            starts.append(env.restore(snapshot))
            for _ in range(16):
                frame, _, _ = env.step(action)
            states.append(env.env.get_sim_state())
            frames.append(frame)
        for i in (1, 2):
            np.testing.assert_array_equal(states[0], states[i])
            np.testing.assert_array_equal(starts[0], starts[i])
            np.testing.assert_array_equal(frames[0], frames[i])
    finally:
        env.close()
