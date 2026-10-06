"""Calibrate fixed visual predicates at their intended temporal reductions.

Training reference banks and encoder remain unchanged. Only original validation
trajectories set the Always-safe and Eventually-reach margins. The artifact is
written and hashed before the original test trajectories are encoded or read.
Physical simulator criteria never change and do not enter runtime scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import classification, sha256, write_json
from .reference_predicates import ReferencePredicates, calibrate_threshold, encoder_feature_hash
from .world_model import load_checkpoint


def temporal_thresholds(raw_safe_traces, raw_reach_traces, physical_safe, physical_reach,
                        goal_safe_raw):
    safe_min = np.asarray([np.min(trace) for trace in raw_safe_traces])
    reach_max = np.asarray([np.max(trace) for trace in raw_reach_traces])
    safe_margin = calibrate_threshold(safe_min, physical_safe, maximum=goal_safe_raw-1e-6)
    goal_threshold = -calibrate_threshold(reach_max, physical_reach, maximum=-1e-6)
    return safe_margin, goal_threshold


def diagnostics(rows, safe_margin, goal_threshold):
    safe_scores = np.asarray([row["raw_safe"].min()-safe_margin for row in rows])
    reach_scores = np.asarray([row["raw_reach"].max()+goal_threshold for row in rows])
    safe_labels = np.asarray([row["safe"].all() for row in rows])
    reach_labels = np.asarray([row["reach"].any() for row in rows])
    return {"trajectory_safety":classification(safe_scores,safe_labels),
            "trajectory_reach":classification(reach_scores,reach_labels),
            "trajectory_joint":classification(np.minimum(safe_scores,reach_scores),safe_labels&reach_labels),
            "frame_safety":classification(np.concatenate([r["raw_safe"] for r in rows])-safe_margin,
                                          np.concatenate([r["safe"] for r in rows])),
            "frame_reach":classification(np.concatenate([r["raw_reach"] for r in rows])+goal_threshold,
                                         np.concatenate([r["reach"] for r in rows])),
            "episodes":[{"id":row["id"],"policy":row["policy"],"physical_safe":bool(safe),
                         "physical_reach":bool(reach),"always_safe":float(safe_score),
                         "eventually_reach":float(reach_score),"stl":float(min(safe_score,reach_score))}
                        for row,safe,reach,safe_score,reach_score in zip(rows,safe_labels,reach_labels,safe_scores,reach_scores)]}


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    refs=ReferencePredicates.from_artifact(args.references,args.device)
    old_margin,old_goal=refs.safe_margin,refs.goal_threshold
    cache=torch.load(args.cache,weights_only=True,map_location="cpu")
    data_hash,checkpoint_hash=sha256(args.dataset),sha256(args.checkpoint)
    if cache["dataset_sha256"]!=data_hash or refs.metadata["dataset_sha256"]!=data_hash:
        raise ValueError("Dataset hashes disagree")
    if cache["encoder_checkpoint_sha256"]!=checkpoint_hash or refs.metadata["checkpoint_sha256"]!=checkpoint_hash:
        raise ValueError("Cached encoder checkpoint provenance differs")
    if any(row["split"] not in ("train","val") for row in cache["episodes"].values()):
        raise ValueError("Calibration cache unexpectedly contains test episodes")
    goal=cache["goal"].to(args.device)
    goal_safe_raw=float(refs.raw_safe(goal))
    def row(ep,key,z):
        z=z.to(args.device).float()
        return {"id":key,"policy":str(ep.attrs["policy"]),
                "raw_safe":refs.raw_safe(z).cpu().numpy(),
                "raw_reach":(F.cosine_similarity(z,goal,dim=-1)-1).cpu().numpy(),
                "safe":ep["diagnostics/safe"][:].astype(bool),
                "reach":ep["diagnostics/reach"][:].astype(bool)}
    rows={"val":[]}
    with h5py.File(args.dataset) as data:
        for key,cached in cache["episodes"].items():
            if cached["split"]!="val":continue
            ep=data["trajectories"][key]
            if ep.attrs["split"]!="val" or len(ep["rgb"])!=len(cached["z"]):
                raise ValueError("Cache split/length does not match original dataset")
            rows["val"].append(row(ep,key,cached["z"]))
    refs.safe_margin,refs.goal_threshold=temporal_thresholds(
        [r["raw_safe"] for r in rows["val"]],[r["raw_reach"] for r in rows["val"]],
        [r["safe"].all() for r in rows["val"]],[r["reach"].any() for r in rows["val"]],goal_safe_raw)
    refs.metadata={**refs.metadata,
        "temporal_recalibration":{"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "source_reference_artifact":str(Path(args.references).resolve()),"source_reference_sha256":sha256(args.references),
            "cache":str(Path(args.cache).resolve()),"cache_sha256":sha256(args.cache),
            "calibration_episodes":[r["id"] for r in rows["val"]],"calibration_split":"val",
            "calibration_frame_counts":[len(r["raw_safe"]) for r in rows["val"]],
            "old_safe_margin":old_margin,"old_goal_threshold":old_goal,
            "new_safe_margin":refs.safe_margin,"new_goal_threshold":refs.goal_threshold,
            "selection_rule":"Max validation balanced accuracy separately on min_t raw_safe vs Always physical_safe and max_t raw_reach vs Eventually physical_reach; ties favor specificity",
            "bank_or_encoder_changed":False,"test_used_for_fitting":False,
            "physical_criteria_changed":False,
            "limitation":"Calibrated on 97-frame traces, not a formal guarantee at other lengths or shifted mug/camera distributions"}}
    # Freeze before reading/encoding any test episode. Never overwrite this file.
    refs.save(output)
    frozen_hash=sha256(output)
    write_json(output.parent/"frozen_before_test.json",{"reference_sha256":frozen_hash,
        "safe_margin":refs.safe_margin,"goal_threshold":refs.goal_threshold,
        "calibration_split":"val","test_read_yet":False})
    shutil.copy2(Path(__file__),output.parent/"source_temporal_calibration.py")
    print(f"FROZEN {output}: safe_margin={refs.safe_margin}, goal_threshold={refs.goal_threshold}",flush=True)
    model,_=load_checkpoint(args.checkpoint,args.device)
    if encoder_feature_hash(model)!=refs.metadata["encoder_feature_sha256"]:
        raise ValueError("Representation hash mismatch")
    rows["test"]=[]
    with h5py.File(args.dataset) as data:
        for key,ep in data["trajectories"].items():
            if ep.attrs["split"]!="test":continue
            rgb=torch.as_tensor(ep["rgb"][:],device=args.device).permute(0,3,1,2)
            with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.float16,
                                enabled=str(args.device).startswith("cuda")):
                z=torch.cat([model.encode(batch).float() for batch in rgb.split(128)])
            rows["test"].append(row(ep,key,z))
    if sha256(output)!=frozen_hash:raise RuntimeError("Frozen reference artifact was modified")
    report={"artifact":str(output.resolve()),"artifact_sha256":frozen_hash,
            "safe_margin":refs.safe_margin,"goal_threshold":refs.goal_threshold,
            "old_safe_margin":old_margin,"old_goal_threshold":old_goal,
            "goal_reference_safe_margin":goal_safe_raw-refs.safe_margin,
            "frozen_before_test":True,"original_reference_changed":False,"splits":{}}
    for split,examples in rows.items():
        report["splits"][split]={"old_frame_calibration":diagnostics(examples,old_margin,old_goal),
                                 "new_temporal_calibration":diagnostics(examples,refs.safe_margin,refs.goal_threshold)}
    write_json(output.with_suffix(".json"),report)
    for split,values in report["splits"].items():
        for version,result in values.items():
            print(split,version,{k:v for k,v in result.items() if k.startswith("trajectory_")},flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","references","dataset","cache","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="cuda")
    run(p.parse_args())


if __name__=="__main__":main()
