from argparse import Namespace
import json

import h5py
import pytest

from scripts.launch_visual_campaign import prepare


def inputs(tmp_path, seed=9000):
    dataset = tmp_path / "data.h5"
    with h5py.File(dataset, "w") as h:
        h.attrs["complete"] = True
        episodes = h.create_group("trajectories")
        for i, split in enumerate(("train", "val", "test")):
            ep = episodes.create_group(str(i))
            ep.attrs["split"] = split
            ep.attrs["scene_seed"] = 100 + i
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"manifest test only; not loaded as a model")
    return Namespace(dataset=dataset, checkpoint=checkpoint, output=tmp_path / "campaign",
                     banks_per_wave=[2, 3], seed=seed, prepare_only=True)


def test_prepared_campaign_has_disjoint_waves_and_does_not_launch(tmp_path):
    output = prepare(inputs(tmp_path))
    protocol = json.loads((output / "protocol.json").read_text())
    seeds = [{w["seed"] + i for i in range(w["banks"])} for w in protocol["waves"]]
    assert not seeds[0] & seeds[1]
    assert protocol["fresh_trajectories"] == 8 * 5 * 9
    assert (output / "source_snapshot/scripts/run_visual_campaign.py").is_file()
    assert not (output / "supervisor.pid").exists()


def test_launcher_rejects_source_scene_leakage(tmp_path):
    args = inputs(tmp_path, seed=100)
    with pytest.raises(ValueError, match="overlap"):
        prepare(args)
    assert not args.output.exists()


def test_launcher_does_not_overwrite_experiment(tmp_path):
    args = inputs(tmp_path)
    args.output.mkdir()
    marker = args.output / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        prepare(args)
    assert marker.read_text() == "existing evidence"
