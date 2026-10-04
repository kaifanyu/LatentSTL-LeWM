"""Leakage-resistant trajectory windows. Only RGB and actions reach training."""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryWindowDataset(Dataset):
    def __init__(self, path: str | Path, split: str = "train", context: int = 3,
                 horizon: int = 1):
        self.path = str(Path(path).resolve())
        self.split, self.context, self.horizon = split, int(context), int(horizon)
        if context < 1 or horizon < 1:
            raise ValueError("context and horizon must be positive")
        self.frames = self.context + self.horizon
        self.index: list[tuple[str, int]] = []
        self.trajectory_ids: list[str] = []
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError("Dataset is incomplete; wait for atomic trajectories.h5 publication")
            self.metadata = dict(handle.attrs)
            for key, episode in handle["trajectories"].items():
                if episode.attrs["split"] != split:
                    continue
                if len(episode["rgb"]) != len(episode["action"]) + 1:
                    raise ValueError(f"Misaligned RGB/action trajectory: {key}")
                self.trajectory_ids.append(key)
                self.index.extend((key, t) for t in range(len(episode["rgb"]) - self.frames + 1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        # h5py handles must be opened independently after DataLoader forks.
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        key, t = self.index[index]
        episode = self._file["trajectories"][key]
        rgb = np.asarray(episode["rgb"][t:t + self.frames], dtype=np.float32) / 255.0
        action = np.asarray(episode["action"][t:t + self.frames - 1], dtype=np.float32)
        return {"rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(),
                "action": torch.from_numpy(action)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"], state["_pid"] = None, None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def training_action_statistics(path: str | Path):
    """Raw training actions only; validation and test never affect scaling."""
    with h5py.File(path, "r") as handle:
        actions = [np.asarray(ep["action"], dtype=np.float64)
                   for ep in handle["trajectories"].values() if ep.attrs["split"] == "train"]
    if not actions:
        raise ValueError("Dataset has no training trajectories")
    values = np.concatenate(actions)
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 0.1).astype(np.float32)


def audit_dataset(path: str | Path, videos_dir: str | Path | None = None):
    """Check real collection alignment, bounds, candidate resets, and split isolation."""
    import hashlib
    from collections import Counter

    banks, start_splits, counts, coverage, selected = {}, {}, Counter(), Counter(), {}
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError("Dataset is not complete")
        action_dim = int(handle.attrs["action_dim"])
        low, high = handle.attrs["action_low"], handle.attrs["action_high"]
        for key, ep in handle["trajectories"].items():
            actions = ep["action"][:]
            if actions.ndim != 2 or actions.shape[1] != action_dim or len(ep["rgb"]) != len(actions) + 1:
                raise ValueError(f"Misaligned trajectory {key}")
            if ep["rgb"].dtype != np.uint8 or not np.isfinite(actions).all():
                raise ValueError(f"Invalid RGB/actions in {key}")
            if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
                raise ValueError(f"Action bounds violated in {key}")
            split, bank = str(ep.attrs["split"]), str(ep.attrs["bank_id"])
            start_hash = str(ep.attrs["start_state_sha256"])
            image_hash = hashlib.sha256(ep["rgb"][0].tobytes()).hexdigest()
            entry = (split, start_hash, image_hash)
            if bank in banks and banks[bank] != entry:
                raise ValueError(f"Candidate bank {bank} differs in split, initial state, or initial RGB")
            if start_hash in start_splits and start_splits[start_hash] != split:
                raise ValueError(f"Shared start state leaks across splits: {bank}")
            banks[bank], start_splits[start_hash] = entry, split
            counts[split] += 1
            counts["transitions"] += len(actions)
            diagnostics = ep["diagnostics"]
            safe, reach = bool(diagnostics["safe"][:].all()), bool(diagnostics["reach"][:].any())
            label = f"safe={safe},reach={reach}"
            coverage[label] += 1
            selected.setdefault(label, key)
        result = {"complete": True, "counts": dict(counts), "num_banks": len(banks),
                  "physical_coverage": dict(coverage), "representative_trajectories": selected,
                  "same_start_bank_rgb_identical": True, "split_start_states_disjoint": True,
                  "raw_actions_within_bounds": True}
        if videos_dir is not None:
            import imageio.v2 as imageio
            output = Path(videos_dir)
            output.mkdir(parents=True, exist_ok=False)
            for label, key in selected.items():
                filename = label.replace("=", "_").replace(",", "_") + f"_{key}.mp4"
                with imageio.get_writer(output / filename, fps=int(handle.attrs["frame_freq_hz"]), codec="libx264", quality=8) as writer:
                    for frame in handle["trajectories"][key]["rgb"]:
                        writer.append_data(frame)
            result["videos_dir"] = str(output.resolve())
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Audit completed LIBERO data and export diagnostic videos")
    parser.add_argument("path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--videos-dir", type=Path)
    args = parser.parse_args()
    report = audit_dataset(args.path, args.videos_dir)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))
