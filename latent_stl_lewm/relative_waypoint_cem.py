"""Initial-image-relative waypoint experiment for static scene changes.

This assumes the initial scene difference is approximately an additive offset
in normalized visual feature space. It is an experimental approximation, not an
object detector. The safety reference banks remain unchanged in this variant.
No simulator state or physical success information enters alignment or CEM.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F

from . import waypoint_cem
from .evaluate import sha256, write_json


class RelativeWaypointGuide(waypoint_cem.WaypointGuide):
    def start(self, latent):
        selected = super().start(latent)
        self.initial_offset = F.normalize(latent.reshape(-1).float(), dim=0) - F.normalize(self.path[0], dim=0)
        selected["relative_initial_offset_norm"] = float(self.initial_offset.norm())
        selected["alignment_assumption"] = "Static initial-scene difference is approximately additive in unit-normalized latent features"
        return selected

    def update(self, latent, elapsed_steps):
        observed = F.normalize(latent.reshape(-1).float(), dim=0) - self.initial_offset
        alignment = super().update(observed, elapsed_steps)
        alignment["relative_initial_alignment"] = True
        return alignment

    @property
    def target(self):
        return F.normalize(super().target, dim=0) + self.initial_offset


def run_relative(args):
    if args.scenario == "baseline":
        # Baseline is allowed as an explicit ablation; this is not automatically
        # selected based on privileged knowledge of whether the mug moved.
        pass
    original = waypoint_cem.WaypointGuide
    waypoint_cem.WaypointGuide = RelativeWaypointGuide
    try:
        return waypoint_cem.run(args)
    finally:
        waypoint_cem.WaypointGuide = original
        output = Path(args.output)
        if output.exists():
            saved = output / "source_snapshot" / Path(__file__).name
            shutil.copy2(__file__, saved)
            config_path = output / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                config["relative_waypoint_experiment"] = {
                    "formula": "offset=normalize(initial_RGB_latent)-normalize(training_start); match normalize(current)-offset to training path; target normalize(training_waypoint)+offset",
                    "safety_banks_shifted": False,
                    "runtime_additional_input": "initial RGB latent only",
                    "source_sha256": sha256(saved),
                }
                config["source_hashes"][saved.name] = sha256(saved)
                write_json(config_path, config)
                manifest = output / "source_snapshot" / "manifest.json"
                write_json(manifest, config["source_hashes"])


if __name__ == "__main__":
    run_relative(waypoint_cem.parser().parse_args())
