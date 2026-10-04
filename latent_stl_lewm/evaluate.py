"""Held-out Levels 0/1/2 evaluation, without using physical state to score actions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import warnings
import h5py
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr, pearsonr
from .logic import PredicateConfig, latent_distance, score_trajectories


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def classification(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    predicted = scores >= 0
    positives, negatives = int(labels.sum()), int((~labels).sum())
    recall = float(predicted[labels].mean()) if positives else None
    specificity = float((~predicted[~labels]).mean()) if negatives else None
    # Rank-sum AUC assigns half credit to ties, including collapsed scores.
    auc = float((rankdata(scores)[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
    return {"n": len(labels), "positives": positives, "negatives": negatives,
            "sign_agreement": float((predicted == labels).mean()) if len(labels) else None,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "auc": auc, "false_safe_rate": float(predicted[~labels].mean()) if negatives else None,
            "false_unsafe_rate": float((~predicted[labels]).mean()) if positives else None,
            "false_safe_count": int((predicted & ~labels).sum()), "false_unsafe_count": int((~predicted & labels).sum())}


def agreement(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    valid = len(predicted) > 2 and np.ptp(predicted) > 1e-10 and np.ptp(actual) > 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(pearsonr(predicted, actual).statistic) if valid else None
        spearman = float(spearmanr(predicted, actual).statistic) if valid else None
    return {**classification(predicted, actual >= 0), "pearson": pearson, "spearman": spearman,
            "mean_absolute_robustness_error": float(np.abs(predicted - actual).mean()) if len(actual) else None,
            "actual_robustness_std": float(np.std(actual)), "predicted_robustness_std": float(np.std(predicted))}


def _attr(value):
    return value.decode() if isinstance(value, bytes) else value


@torch.no_grad()
def encode_rgb(model, rgb, device, batch_size=32):
    tensors = []
    for offset in range(0, len(rgb), batch_size):
        batch = torch.as_tensor(np.asarray(rgb[offset:offset + batch_size]), device=device).permute(0, 3, 1, 2)
        tensors.append(model.encode(batch).float().cpu())
    return torch.cat(tensors)


def _labels(diag, end=None):
    sl = slice(0, end)
    if "safe" not in diag or "reach" not in diag:
        return None
    return {"safe": bool(np.asarray(diag["safe"])[sl].all()), "reach": bool(np.asarray(diag["reach"])[sl].any()),
            "terminal_reach": bool(np.asarray(diag["reach"])[sl][-1])}


def evaluate(checkpoint, dataset, output, *, device="cpu", split="test", max_trajectories=48,
             horizons=(1, 2, 4, 8), predicate_config=None, ranking_horizon=8, seed=0):
    from .world_model import load_checkpoint
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = predicate_config or PredicateConfig()
    if config.end is not None:
        raise ValueError("Evaluation uses the full observed window at each horizon; leave end=None")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, checkpoint_data = load_checkpoint(checkpoint, device=device)
    model.eval()
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], capture_output=True, text=True).stdout.strip() or None
    git_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parents[1], capture_output=True, text=True).stdout.strip()
    config_record = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
                     "dataset": str(Path(dataset).resolve()), "dataset_sha256": sha256(dataset),
                     "predicate": asdict(config), "formula": "Always_[0,H](preserve) AND Eventually_[0,H](reach)",
                     "threshold_source": "Explicit CLI/config values; no physical labels used to fit thresholds",
                     "split": split, "max_trajectories": max_trajectories, "horizons": list(horizons),
                     "ranking_horizon": ranking_horizon, "seed": seed, "device": str(device),
                     "git_commit": git_commit, "git_dirty": git_dirty, "model_config": checkpoint_data.get("config"),
                     "encoder_checkpoint_sha256": sha256(checkpoint)}
    trajectories = []
    with h5py.File(dataset, "r") as data:
        refs = data["references"]
        goal_rgb, safe_rgb = np.asarray(refs["goal_rgb"]), np.asarray(refs["safe_rgb"])
        config_record["goal_image_sha256"] = hashlib.sha256(goal_rgb.tobytes()).hexdigest()
        config_record["safe_image_sha256"] = hashlib.sha256(safe_rgb.tobytes()).hexdigest()
        goal, safe = encode_rgb(model, np.stack([goal_rgb, safe_rgb]), device)
        eligible = [name for name, group in data["trajectories"].items() if _attr(group.attrs.get("split", "")) == split]
        # Preserve complete same-start banks when limiting the evaluation set.
        banks, independent = {}, []
        for name in sorted(eligible):
            bank = str(_attr(data["trajectories"][name].attrs.get("bank_id", "")))
            if bank and bank not in ("-1", "None", "none"):
                banks.setdefault(bank, []).append(name)
            else:
                independent.append(name)
        selected = []
        for names in banks.values():
            if len(selected) + len(names) <= max_trajectories:
                selected.extend(names)
        selected += independent[:max(0, max_trajectories - len(selected))]
        for name in selected:
            group = data["trajectories"][name]
            rgb = group["rgb"]
            trajectories.append({"id": name, "z": encode_rgb(model, rgb, device), "actions": np.asarray(group["action"]),
                                 "bank_id": str(_attr(group.attrs.get("bank_id", ""))),
                                 "initial_rgb_sha256": hashlib.sha256(np.asarray(rgb[0]).tobytes()).hexdigest(),
                                 "diagnostics": {key: np.asarray(val) for key, val in group.get("diagnostics", {}).items()},
                                 "metadata": {key: _attr(val) for key, val in group.attrs.items()}})
    if not trajectories:
        raise ValueError(f"No complete {split!r} trajectories/banks fit max_trajectories={max_trajectories}")
    config_record["trajectory_ids"] = [item["id"] for item in trajectories]
    config_record["config_sha256"] = hashlib.sha256(json.dumps(json_safe(config_record), sort_keys=True).encode()).hexdigest()
    write_json(output / "config.json", config_record)
    goal, safe = goal.to(device), safe.to(device)
    distance_kwargs = dict(metric=config.metric, feature_ndim=config.feature_ndim)
    rows0, by_horizon, ranking = [], {str(h): {"pairs": [], "errors": [], "persistence_errors": [], "shuffled_errors": [], "action_sensitivity": []} for h in horizons}, {}
    action_pool = torch.as_tensor(np.concatenate([row["actions"] for row in trajectories]), device=device, dtype=torch.float32)
    for trajectory in trajectories:
        z = trajectory["z"].to(device)
        scores = score_trajectories(z[None], goal, safe, config)
        labels = _labels(trajectory["diagnostics"])
        rows0.append({"id": trajectory["id"], "metadata": trajectory["metadata"], "labels": labels,
                      "scores": {key: val.cpu().tolist()[0] for key, val in scores.items()}})
        for horizon in horizons:
            if len(z) <= horizon:
                continue
            starts = np.unique(np.linspace(0, len(z) - horizon - 1, min(4, len(z) - horizon), dtype=int))
            actions = torch.as_tensor(np.stack([trajectory["actions"][start:start+horizon] for start in starts]), device=device, dtype=torch.float32)
            initial = z[torch.as_tensor(starts, device=device)]
            with torch.no_grad():
                predicted = model.rollout(initial, actions).float()
                shuffled_actions = action_pool[torch.randint(len(action_pool), actions.shape[:2], device=device)]
                shuffled = model.rollout(initial, shuffled_actions).float()
            actual = torch.stack([z[start:start+horizon+1] for start in starts])
            pred_scores = score_trajectories(predicted, goal, safe, config)
            true_scores = score_trajectories(actual, goal, safe, config)
            bucket = by_horizon[str(horizon)]
            for metric in ("cosine", "normalized_l2", "l2"):
                error = latent_distance(predicted[:, -1], actual[:, -1], metric, feature_ndim=config.feature_ndim)
                baseline = latent_distance(initial, actual[:, -1], metric, feature_ndim=config.feature_ndim)
                shuffled_error = latent_distance(shuffled[:, -1], actual[:, -1], metric, feature_ndim=config.feature_ndim)
                sensitivity = latent_distance(predicted[:, -1], shuffled[:, -1], metric, feature_ndim=config.feature_ndim)
                bucket["shuffled_errors"].extend({"metric": metric, "value": float(value)} for value in shuffled_error.cpu())
                bucket["action_sensitivity"].extend({"metric": metric, "value": float(value)} for value in sensitivity.cpu())
                bucket["errors"].extend({"metric": metric, "value": float(value)} for value in error.cpu())
                bucket["persistence_errors"].extend({"metric": metric, "value": float(value)} for value in baseline.cpu())
            for idx, start in enumerate(starts):
                bucket["pairs"].append({"trajectory": trajectory["id"], "start": int(start),
                                        "predicted": float(pred_scores["stl"][idx]), "actual": float(true_scores["stl"][idx]),
                                        "predicted_reach_trace": pred_scores["reach_trace"][idx].cpu().tolist(),
                                        "predicted_preserve_trace": pred_scores["preserve_trace"][idx].cpu().tolist(),
                                        "actual_reach_trace": true_scores["reach_trace"][idx].cpu().tolist(),
                                        "actual_preserve_trace": true_scores["preserve_trace"][idx].cpu().tolist()})
        bank = trajectory["bank_id"]
        if bank and bank not in ("-1", "None", "none") and len(z) > ranking_horizon:
            with torch.no_grad():
                actions = torch.as_tensor(trajectory["actions"][:ranking_horizon], device=device, dtype=torch.float32)[None]
                predicted = model.rollout(z[:1], actions)
            pred_scores = score_trajectories(predicted, goal, safe, config)
            actual_scores = score_trajectories(z[:ranking_horizon+1][None], goal, safe, config)
            ranking.setdefault(bank, []).append({"id": trajectory["id"], "initial_rgb_sha256": trajectory["initial_rgb_sha256"],
                                                 "predicted": {name: float(pred_scores[name][0]) for name in ("terminal", "eventual", "stl")},
                                                 "actual": {name: float(actual_scores[name][0]) for name in ("terminal", "eventual", "stl")},
                                                 "labels": _labels(trajectory["diagnostics"], ranking_horizon+1),
                                                 "diagnostics": {key: val[:ranking_horizon+1].tolist() for key,val in trajectory["diagnostics"].items()},
                                                 "actions": trajectory["actions"][:ranking_horizon].tolist()})
    labelled = [row for row in rows0 if row["labels"] is not None]
    level0 = {"trajectories": len(rows0), "labelled_trajectories": len(labelled), "thresholds_calibrated": False}
    if labelled:
        joint_labels = [row["labels"]["safe"] and row["labels"]["reach"] for row in labelled]
        level0["stl_vs_physical_joint"] = classification([row["scores"]["stl"] for row in labelled], joint_labels)
        level0["preserve_vs_physical_safe"] = classification([min(row["scores"]["preserve_trace"]) for row in labelled], [row["labels"]["safe"] for row in labelled])
        level0["eventual_vs_physical_reach"] = classification([row["scores"]["eventual"] for row in labelled], [row["labels"]["reach"] for row in labelled])
        level0["terminal_vs_physical_joint"] = classification([row["scores"]["terminal"] + config.goal_threshold for row in labelled], joint_labels)
        level0["physical_outcome_counts"] = {f"safe={safe_label},reach={reach_label}": sum(row["labels"]["safe"] == safe_label and row["labels"]["reach"] == reach_label for row in labelled) for safe_label in (False, True) for reach_label in (False, True)}
    level1 = {}
    for horizon, bucket in by_horizon.items():
        if not bucket["pairs"]:
            level1[horizon] = {"status": "insufficient trajectory length"}
            continue
        level1[horizon] = {"robustness": agreement([row["predicted"] for row in bucket["pairs"]], [row["actual"] for row in bucket["pairs"]]),
                           "prediction_error": {metric: float(np.mean([row["value"] for row in bucket["errors"] if row["metric"] == metric])) for metric in ("cosine", "normalized_l2", "l2")},
                           "shuffled_action_error": {metric: float(np.mean([row["value"] for row in bucket["shuffled_errors"] if row["metric"] == metric])) for metric in ("cosine", "normalized_l2", "l2")},
                           "action_sensitivity": {metric: float(np.mean([row["value"] for row in bucket["action_sensitivity"] if row["metric"] == metric])) for metric in ("cosine", "normalized_l2", "l2")},
                           "persistence_error": {metric: float(np.mean([row["value"] for row in bucket["persistence_errors"] if row["metric"] == metric])) for metric in ("cosine", "normalized_l2", "l2")}}
    level2 = {"horizon": ranking_horizon, "evaluation_scope": "Prefix ranking only; this does not establish full-task planning success", "banks": {}}
    for bank, rows in ranking.items():
        if len(rows) < 3 or len({row["initial_rgb_sha256"] for row in rows}) != 1:
            level2["banks"][bank] = {"status": "invalid: needs >=3 candidates with identical initial RGB", "candidates": len(rows)}
            continue
        results = {}
        actual_stl = np.asarray([row["actual"]["stl"] for row in rows])
        for objective in ("terminal", "eventual", "stl"):
            predicted_scores = np.asarray([row["predicted"][objective] for row in rows])
            actual_scores = np.asarray([row["actual"][objective] for row in rows])
            selected = int(np.argmax(predicted_scores))
            topk = min(5, max(1, len(rows) // 4))
            pred_top = set(np.argsort(-predicted_scores, kind="stable")[:topk])
            true_top = set(np.argsort(-actual_scores, kind="stable")[:topk])
            diag = rows[selected]["diagnostics"]
            results[objective] = {"agreement": agreement(predicted_scores, actual_scores), "selected_id": rows[selected]["id"],
                                  "top1_regret": float(actual_scores.max() - actual_scores[selected]),
                                  "true_stl_of_selected": float(actual_stl[selected]),
                                  "true_stl_regret": float(actual_stl.max() - actual_stl[selected]),
                                  "top_k": topk, "top_k_recall": len(pred_top & true_top) / topk,
                                  "selected_physical_labels": rows[selected]["labels"],
                                  "max_object_displacement": float(np.max(diag["object_displacement"])) if "object_displacement" in diag else None,
                                  "max_contact_count": float(np.max(diag["contact_count"])) if "contact_count" in diag else None}
        level2["banks"][bank] = {"status": "evaluated", "candidates": len(rows), "objectives": results}
    valid_banks = [bank for bank in level2["banks"].values() if bank["status"] == "evaluated"]
    level2["status"] = "evaluated" if valid_banks else "not evaluated: no complete same-start candidate banks supplied"
    level2["valid_banks"] = len(valid_banks)
    level2["summary"] = {}
    for objective in ("terminal", "eventual", "stl"):
        records = [bank["objectives"][objective] for bank in valid_banks]
        correlations = [row["agreement"]["spearman"] for row in records if row["agreement"]["spearman"] is not None]
        labels = [row["selected_physical_labels"] for row in records if row["selected_physical_labels"] is not None]
        level2["summary"][objective] = {"mean_spearman": float(np.mean(correlations)) if correlations else None,
                                        "mean_true_stl_regret": float(np.mean([row["true_stl_regret"] for row in records])) if records else None,
                                        "safe_selection_rate": float(np.mean([row["safe"] for row in labels])) if labels else None,
                                        "goal_success_rate": float(np.mean([row["reach"] for row in labels])) if labels else None,
                                        "constraint_violation_rate": float(np.mean([not row["safe"] for row in labels])) if labels else None}
    all_z = torch.cat([row["z"] for row in trajectories]).float()
    flattened = all_z.flatten(1)
    latent_std = float(flattened.std(0, unbiased=False).mean())
    norm = float(flattened.norm(dim=-1).mean())
    diagnostic = {"latent_mean_feature_std": latent_std, "latent_mean_norm": norm,
                  "relative_latent_variation": latent_std / max(norm / np.sqrt(flattened.shape[-1]), 1e-12),
                  "goal_safe_distance": float(latent_distance(goal, safe, **distance_kwargs)),
                  "spatial_tokens_available": config.feature_ndim == 2,
                  "notes": ["Physical labels are diagnostic only; neither thresholds nor candidate scores were fitted to them.",
                            "Whole-image preservation may penalize desired arm motion; this global encoder has no object mask.",
                            "Overlapping prefixes are not independent samples; correlations are descriptive, without confidence claims.",
                            "Short raw-action horizons can cover only fractions of a second and cannot certify the full reaching task.",
                            "Shuffled-action baseline resamples actions from the held-out action pool; action sensitivity is the distance between correct- and shuffled-action predictions.",
                            "Thresholds are initial explicit tolerances; validation is required before claiming calibrated predicates."]}
    failure_reasons = []
    joint = level0.get("stl_vs_physical_joint", {})
    if not joint or joint.get("balanced_accuracy") is None:
        failure_reasons.append("Level 0 inconclusive: missing labels or physical outcome diversity")
    elif joint["balanced_accuracy"] < 0.65:
        failure_reasons.append("Level 0 predicate/representation failure: joint balanced accuracy < 0.65")
    for name in ("preserve_vs_physical_safe", "eventual_vs_physical_reach"):
        result = level0.get(name, {})
        if result.get("balanced_accuracy") is None or result["balanced_accuracy"] < 0.65:
            failure_reasons.append(f"Level 0 component predicate unvalidated: {name} needs both label classes and balanced accuracy >=0.65")
    if diagnostic["relative_latent_variation"] < 0.01:
        failure_reasons.append("Representation collapse warning: relative latent variation < 0.01")
    credible_horizons = [int(h) for h, record in level1.items() if record.get("robustness", {}).get("spearman") is not None and record["robustness"]["spearman"] >= 0.3 and record["robustness"]["positives"] >= 3 and record["robustness"]["negatives"] >= 3 and record["prediction_error"]["cosine"] < record["persistence_error"]["cosine"] and record["prediction_error"]["cosine"] < 0.99 * record["shuffled_action_error"]["cosine"]]
    if not credible_horizons:
        failure_reasons.append("Level 1: no horizon passed rank correlation >=0.3, both sign classes, and improvement over persistence/shuffled actions")
    rank_corr = level2["summary"]["stl"]["mean_spearman"]
    if len(valid_banks) < 3 or rank_corr is None or rank_corr < 0.3:
        failure_reasons.append("Level 2 insufficient: need >=3 same-start banks and mean Spearman >=0.3")
    if ranking_horizon not in credible_horizons:
        failure_reasons.append("Ranking horizon is not supported by Level 1 prediction evidence")
    report = {"config_sha256": config_record["config_sha256"], "checkpoint_sha256": config_record["checkpoint_sha256"],
              "dataset_sha256": config_record["dataset_sha256"], "level0": level0, "level1": level1, "level2": level2,
              "diagnostics": diagnostic, "mpc_gate": {"ready": not failure_reasons, "reasons": failure_reasons,
              "credible_horizons": credible_horizons, "maximum_execute_horizon": 4,
              "description": "Conservative engineering gate, not a statistical scientific acceptance test"}}
    write_json(output / "report.json", report)
    write_json(output / "level0_trajectories.json", rows0)
    write_json(output / "level1_pairs.json", by_horizon)
    write_json(output / "level2_candidates.json", ranking)
    _plots(output, rows0, by_horizon, level1)
    _markdown(output, report)
    return report


def _plots(output, rows, horizons, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for row in rows[:8]:
        axes[0].plot(row["scores"]["reach_trace"], label=row["id"], alpha=.7)
        axes[1].plot(row["scores"]["preserve_trace"], alpha=.7)
    for ax, title in zip(axes, ("Reach robustness (real encoded RGB)", "Preserve robustness (real encoded RGB)")):
        ax.axhline(0, color="black", linestyle="--"); ax.set_ylabel(title)
    axes[1].set_xlabel("Observation / control step")
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(output / "predicate_traces.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, len(horizons), figsize=(4*len(horizons), 4), squeeze=False)
    for ax, (horizon, bucket) in zip(axes.flat, horizons.items()):
        actual, pred = [row["actual"] for row in bucket["pairs"]], [row["predicted"] for row in bucket["pairs"]]
        ax.scatter(pred, actual, s=12, alpha=.5)
        ax.axhline(0, color="gray", linewidth=.6); ax.axvline(0, color="gray", linewidth=.6)
        ax.set_title(f"H={horizon}"); ax.set_xlabel("Predicted STL robustness"); ax.set_ylabel("Real STL robustness")
    fig.tight_layout(); fig.savefig(output / "robustness_agreement.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = [(int(h), record) for h, record in summary.items() if "prediction_error" in record]
    if valid:
        ax.plot([h for h, _ in valid], [record["prediction_error"]["cosine"] for _, record in valid], "o-", label="LeWM prediction")
        ax.plot([h for h, _ in valid], [record["persistence_error"]["cosine"] for _, record in valid], "o-", label="Persistence baseline")
        ax.legend()
    ax.set_xlabel("Prediction horizon"); ax.set_ylabel("Cosine distance to actual latent")
    fig.tight_layout(); fig.savefig(output / "prediction_horizons.png", dpi=150); plt.close(fig)


def _markdown(output, report):
    lines = ["# Held-out latent-STL evaluation", "", f"Checkpoint SHA256: `{report['checkpoint_sha256']}`", "",
             "The optimizer sees RGB latents and actions. Physical state appears only in held-out diagnostics.", "",
             f"Closed-loop gate: **{'ready' if report['mpc_gate']['ready'] else 'not ready'}**.", ""]
    lines += [f"- {reason}" for reason in report["mpc_gate"]["reasons"]]
    lines += ["", "## Prediction and robustness", "", "| Horizon | Cosine error | Persistence error | Spearman | False safe |", "|---|---:|---:|---:|---:|"]
    for horizon, record in report["level1"].items():
        if "robustness" in record:
            lines.append(f"| {horizon} | {record['prediction_error']['cosine']:.5g} | {record['persistence_error']['cosine']:.5g} | {record['robustness']['spearman']} | {record['robustness']['false_safe_rate']} |")
    lines += ["", "## Predicate validation", "", "```json", json.dumps(json_safe(report["level0"]), indent=2), "```", "",
              "## Offline matched candidate ranking", "", "Each objective scores the identical supplied same-start candidate bank.", "", "```json", json.dumps(json_safe(report["level2"]["summary"]), indent=2), "```", "",
              "## Limitations", ""] + [f"- {note}" for note in report["diagnostics"]["notes"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-trajectories", type=int, default=48); parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ranking-horizon", type=int, default=8); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", choices=("cosine", "normalized_l2", "l2", "mean_token_cosine"), default="cosine")
    parser.add_argument("--goal-threshold", type=float, default=.02); parser.add_argument("--preserve-threshold", type=float, default=.01)
    parser.add_argument("--semantics", choices=("exact", "smooth"), default="exact"); parser.add_argument("--tau", type=float, default=.01)
    args = parser.parse_args()
    pred = PredicateConfig(metric=args.metric, goal_threshold=args.goal_threshold, preserve_threshold=args.preserve_threshold, semantics=args.semantics, tau=args.tau)
    report = evaluate(args.checkpoint, args.dataset, args.output, device=args.device, split=args.split,
                      max_trajectories=args.max_trajectories, horizons=args.horizons, ranking_horizon=args.ranking_horizon,
                      predicate_config=pred, seed=args.seed)
    print(json.dumps(json_safe({"output": args.output, "mpc_gate": report["mpc_gate"]}), indent=2))


if __name__ == "__main__":
    main()
