"""LeWM predicate forecast errors on held-out recorded actions; no optimization."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import classification, sha256, write_json
from .visual_predicate_benchmark import temporal_scores
from .world_model import load_checkpoint


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    cfg = json.loads(Path(args.calibration).read_text())
    if cfg["method"] != "global" or cfg["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Forecast scorer and encoder do not match")
    def encode(rgb):
        zs = []
        for begin in range(0, len(rgb), 32):
            x = torch.as_tensor(np.asarray(rgb[begin:begin+32]), device=args.device).permute(0, 3, 1, 2)
            with torch.autocast("cuda", dtype=torch.float16):
                zs.append(model.encode(x).float())
        return torch.cat(zs)
    rows = []
    with h5py.File(args.dataset, "r") as h:
        goal = encode(h["references/goal_rgb"][:][None])[0]
        for key, ep in h["trajectories"].items():
            if ep.attrs["split"] != "test":
                continue
            actual = encode(ep["rgb"])
            actions = torch.as_tensor(ep["action"][:], device=args.device)
            for start in range(0, len(actions)-7, 16):
                # Only observations/actions up to start enter history.
                past_begin = max(0, start - model.config.history_size + 1)
                latent_history = [z[None] for z in actual[past_begin:start+1]]
                action_history = [a[None] for a in actions[past_begin:start]]
                predicted = [actual[start]]
                for step in range(8):
                    action_history.append(actions[start+step][None])
                    length = min(len(latent_history), model.config.history_size)
                    with torch.autocast("cuda", dtype=torch.float16):
                        z = model.predict_next(torch.stack(latent_history[-length:], 1),
                                               torch.stack(action_history[-length:], 1)).float()
                    latent_history.append(z); predicted.append(z[0])
                pred = torch.stack(predicted)
                observed = actual[start:start+9]
                for horizon in (1, 2, 4, 8):
                    for name, c in cfg["operating_points"].items():
                        def score(z):
                            s = (c["safe_threshold"]-(1-F.cosine_similarity(z, actual[0], dim=-1)).clamp_min(0)) / c["safe_scale"]
                            g = (c["goal_threshold"]-(1-F.cosine_similarity(z, goal, dim=-1)).clamp_min(0)) / c["goal_scale"]
                            return temporal_scores(s.cpu().numpy(), g.cpu().numpy())
                        p, a = score(pred[:horizon+1]), score(observed[:horizon+1])
                        sl = ep["diagnostics/safe"][start:start+horizon+1].astype(bool)
                        gl = ep["diagnostics/reach"][start:start+horizon+1].astype(bool)
                        rows.append({"episode": key, "bank": str(ep.attrs["bank_id"]),
                                     "start": start, "horizon": horizon, "operating_point": name,
                                     "predicted": p, "observed": a,
                                     "physical_safe": bool(sl.all()), "physical_success": bool(sl.all() and gl.any()),
                                     "latent_cosine_error": float((1-F.cosine_similarity(pred[horizon], observed[horizon], dim=0)).clamp_min(0))})
            print(json.dumps({"phase": "forecast_audit", "episode": key, "rows": len(rows)}), flush=True)
    summary = {}
    for name in cfg["operating_points"]:
        summary[name] = {}
        for horizon in (1, 2, 4, 8):
            chosen = [r for r in rows if r["operating_point"] == name and r["horizon"] == horizon]
            summary[name][horizon] = {
                "windows": len(chosen),
                "prediction_vs_observed_safety": classification([r["predicted"]["safe_min"] for r in chosen], [r["observed"]["safe_min"] >= 0 for r in chosen]),
                "predicted_vs_physical_safety": classification([r["predicted"]["safe_min"] for r in chosen], [r["physical_safe"] for r in chosen]),
                "observed_vs_physical_safety": classification([r["observed"]["safe_min"] for r in chosen], [r["physical_safe"] for r in chosen]),
                "mean_latent_cosine_error": float(np.mean([r["latent_cosine_error"] for r in chosen]))}
    write_json(out / "report.json", {"kind": "Forecast diagnostic on recorded actions, not CEM or closed-loop control",
               "formula_window": "Local observed/predicted [start,start+H]; not full episode satisfaction",
               "history": "Actual images and actions through start; autoregressive predictions afterward",
               "checkpoint_sha256": sha256(args.checkpoint), "dataset_sha256": sha256(args.dataset),
               "summary": summary, "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "calibration", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
