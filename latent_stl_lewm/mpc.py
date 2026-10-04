"""Evidence-gated, short-action-prefix RGB-only CEM MPC in LIBERO."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
from .evaluate import encode_rgb, json_safe, sha256, write_json
from .logic import PredicateConfig, score_trajectories
from .planner import CEMConfig, plan_cem


def check_evidence(evidence, checkpoint, dataset, horizon, execute_horizon):
    """Fail closed on stale, mismatched, or scientifically unsupported evidence."""
    reasons = list(evidence.get("mpc_gate", {}).get("reasons", []))
    if not evidence.get("mpc_gate", {}).get("ready", False):
        reasons.append("Evaluator has not certified the initial engineering gate")
    if evidence.get("checkpoint_sha256") != sha256(checkpoint):
        reasons.append("Evidence checkpoint hash differs from requested checkpoint")
    if evidence.get("dataset_sha256") != sha256(dataset):
        reasons.append("Evidence dataset hash differs from requested dataset")
    if horizon not in evidence.get("mpc_gate", {}).get("credible_horizons", []):
        reasons.append("Requested planning horizon lacks credible prediction evidence")
    if horizon != evidence.get("level2", {}).get("horizon"):
        reasons.append("Requested horizon differs from the validated same-start ranking horizon")
    if not 1 <= execute_horizon <= min(4, horizon):
        reasons.append("Execute horizon must be between 1 and min(4, planning horizon)")
    return reasons


def run_mpc(checkpoint, dataset, evidence_path, output, *, device="cuda", objective="all", horizon=8,
            execute_horizon=1, cycles=24, population=128, iterations=4, elites=16, seed=0, scene_seed=0, render_gpu=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    reasons = check_evidence(evidence, checkpoint, dataset, horizon, execute_horizon)
    if reasons:
        result = {"status": "skipped: evidence gate not ready", "reasons": reasons,
                  "evidence": str(evidence_path.resolve()), "closed_loop_executed": False}
        write_json(output / "status.json", result)
        return result
    eval_config = json.loads((evidence_path.parent / "config.json").read_text())
    stored_config_hash = eval_config.get("config_sha256")
    config_payload = {key: val for key, val in eval_config.items() if key != "config_sha256"}
    actual_config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
    if stored_config_hash != actual_config_hash or stored_config_hash != evidence.get("config_sha256"):
        raise ValueError("Evidence and evaluation config hashes disagree")
    pred_config = PredicateConfig(**eval_config["predicate"])
    from .simulator import MugPreservationEnv
    from .world_model import load_model
    model = load_model(checkpoint, device=device)
    model.eval()
    with h5py.File(dataset, "r") as data:
        metadata = json.loads(data.attrs["metadata_json"])
        train_actions = [np.asarray(group["action"]) for group in data["trajectories"].values() if str(group.attrs.get("split", "")) == "train"]
        if not train_actions:
            raise ValueError("Need training-split action coverage to bound MPC proposals")
        train_actions = np.concatenate(train_actions)
        observed_low, observed_high = train_actions.min(0), train_actions.max(0)
        rgb_refs = np.stack([np.asarray(data["references"]["goal_rgb"]), np.asarray(data["references"]["safe_rgb"])])
    goal, safe = encode_rgb(model, rgb_refs, device).to(device)
    sim_config = metadata["simulator_config"]
    sim_config["render_gpu"] = render_gpu
    objectives = ("terminal", "eventual", "stl") if objective == "all" else (objective,)
    run_config = {"checkpoint_sha256": sha256(checkpoint), "encoder_checkpoint_sha256": sha256(checkpoint),
                  "dataset_sha256": sha256(dataset), "evidence_sha256": sha256(evidence_path),
                  "evaluation_config_sha256": evidence["config_sha256"], "predicate": asdict(pred_config),
                  "goal_image_sha256": eval_config["goal_image_sha256"], "safe_image_sha256": eval_config["safe_image_sha256"],
                  "seed": seed, "scene_seed": scene_seed, "horizon": horizon, "execute_horizon": execute_horizon,
                  "cycles": cycles, "population": population, "iterations": iterations, "elites": elites,
                  "temporal_history": "Observed RGB latent prefix plus candidate future; past satisfaction/violations persist",
                  "objectives": objectives, "simulator_config": sim_config, "physical_state_role": "diagnostics only"}
    env = MugPreservationEnv(sim_config)
    summaries = {}
    try:
        env.reset(scene_seed)
        initial = env.snapshot()
        run_config["raw_environment_action_low"], run_config["raw_environment_action_high"] = env.action_low.tolist(), env.action_high.tolist()
        planner_low, planner_high = np.maximum(env.action_low, observed_low), np.minimum(env.action_high, observed_high)
        run_config["action_low"], run_config["action_high"] = planner_low.tolist(), planner_high.tolist()
        run_config["action_bound_source"] = "Coordinate-wise training action extrema intersected with environment bounds; no diagnostic inputs. This limits extrapolation but does not guarantee joint action-distribution coverage."
        write_json(output / "config.json", run_config)
        for mode in objectives:
            directory = output / mode
            directory.mkdir()
            frames = [env.restore(initial)]
            diagnostics, plans, all_actions = [env.diagnostics()], [], []
            history_z = encode_rgb(model, np.asarray(frames), device).to(device)
            for cycle in range(cycles):
                z0 = history_z[-1:]
                cem = CEMConfig(horizon=horizon, population=population, iterations=iterations, elites=elites, seed=seed+cycle)
                result = plan_cem(model, z0, goal, safe, planner_low, planner_high,
                                  objective=mode, config=cem, predicate_config=pred_config, history_latents=history_z[None, :-1])
                actions = result.pop("actions").cpu().numpy()
                segment = [frames[-1]]
                for action in actions[:execute_horizon]:
                    rgb, done, _ = env.step(action)
                    frames.append(rgb); segment.append(rgb); all_actions.append(action)
                    diagnostics.append(env.diagnostics())
                    if done:
                        break
                actual_z = encode_rgb(model, np.asarray(segment), device).to(device)
                actual = score_trajectories(actual_z[None], goal, safe, pred_config)
                history_z = torch.cat([history_z, actual_z[1:]], dim=0)
                history_actual = score_trajectories(history_z[None], goal, safe, pred_config)
                plans.append({"cycle": cycle, **result, "optimized_action_sequence": actions.tolist(),
                              "selected_action_prefix": actions[:len(segment)-1].tolist(),
                              "executed_prefix_actual_scores": {key: val.cpu().tolist() for key, val in actual.items()},
                              "observed_history_actual_scores": {key: val.cpu().tolist() for key, val in history_actual.items()},
                              "physical_diagnostic_after_execution": diagnostics[-1]})
                write_json(directory / f"cycle_{cycle:03d}.json", plans[-1])
                if done:
                    break
            actual_full = score_trajectories(encode_rgb(model, np.asarray(frames), device).to(device)[None], goal, safe, pred_config)
            np.savez_compressed(directory / "trajectory.npz", rgb=np.asarray(frames), action=np.asarray(all_actions),
                                **{f"diagnostic_{key}": np.asarray([record[key] for record in diagnostics]) for key in diagnostics[0]})
            summaries[mode] = {"cycles": len(plans), "environment_steps": len(all_actions),
                               "actual_stl_robustness": float(actual_full["stl"][0]),
                               "physical_safe": all(bool(item["safe"]) for item in diagnostics),
                               "physical_reached": any(bool(item["reach"]) for item in diagnostics),
                               "maximum_object_displacement": max(float(item["object_displacement"]) for item in diagnostics),
                               "maximum_object_rotation_deg": max(float(item["object_rotation_deg"]) for item in diagnostics),
                               "final_goal_error": float(diagnostics[-1]["goal_error"])}
            _save_visualizations(directory, frames, plans, diagnostics, metadata["control_freq_hz"])
    finally:
        env.close()
    status = {"status": "completed", "closed_loop_executed": True, "objectives": summaries}
    write_json(output / "status.json", status)
    return status


def _save_visualizations(directory, frames, plans, diagnostics, fps):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        imageio.mimsave(directory / "trajectory.mp4", frames, fps=fps, macro_block_size=1)
    except (ImportError, RuntimeError, ValueError):
        imageio.mimsave(directory / "trajectory.gif", frames, duration=1/fps)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot([plan["score"] for plan in plans], label="Best predicted planning score")
    axes[0].plot([plan["executed_prefix_actual_scores"]["stl"][0] for plan in plans], label="Real STL of executed prefix")
    axes[0].axhline(0, color="gray", linestyle="--"); axes[0].legend(); axes[0].set_xlabel("MPC cycle (different evaluation windows)")
    axes[1].plot([float(item["goal_error"]) for item in diagnostics]); axes[1].set_ylabel("Physical goal error, m")
    axes[2].plot([float(item["object_displacement"]) for item in diagnostics]); axes[2].set_ylabel("Mug displacement, m"); axes[2].set_xlabel("Environment step")
    fig.tight_layout(); fig.savefig(directory / "mpc_progress.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "dataset", "evidence", "output"):
        parser.add_argument(f"--{arg}", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--objective", choices=("all", "terminal", "eventual", "stl"), default="all")
    parser.add_argument("--horizon", type=int, default=8); parser.add_argument("--execute-horizon", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=24); parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=4); parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--scene-seed", type=int, default=0); parser.add_argument("--render-gpu", type=int, default=2)
    args = vars(parser.parse_args()); args["evidence_path"] = args.pop("evidence")
    print(json.dumps(json_safe(run_mpc(**args)), indent=2))


if __name__ == "__main__":
    main()
