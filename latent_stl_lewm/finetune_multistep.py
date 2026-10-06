"""Fine-tune local LeWM dynamics with a frozen, unchanged image representation.

Only the original training split supplies gradients. Fixed shuffled validation
windows select checkpoints on prediction error, never on batch-dependent SIGReg.
Cached RGB encodings are constants, and the predictor runs in inference mode
with gradients enabled, matching the dropout/BatchNorm behavior used by CEM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .train import amp_context, file_hash, git_metadata, jsonable, seed_all, write_json
from .world_model import ROOT, load_checkpoint


def freeze_representation(model):
    model.config.backend = "frozen_encoder_learned_predictor"
    model.freeze_encoder()
    # eval() disables dropout and prevents predictor BN statistics from drifting;
    # it does not disable gradients to predictor/action/projection parameters.
    model.eval()


def representation_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("model.encoder.", "model.projector.", "image_mean", "image_std")):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def autoregressive_predictions(model, latents, actions, context, future_steps):
    """Predict z[context:context+H], retaining exactly aligned action history.

    The initial observed context is z[0:context]; action[t] advances z[t] to
    z[t+1]. Later contexts contain predictions and remain differentiable.
    """
    if not 1 <= context <= model.config.history_size or future_steps < 1:
        raise ValueError("Invalid context or future_steps")
    if latents.ndim != 3 or actions.ndim != 3 or latents.shape[0] != actions.shape[0]:
        raise ValueError("Latents/actions need matching batch dimensions")
    if latents.shape[1] < context or actions.shape[1] != context + future_steps - 1:
        raise ValueError("Need context + future_steps - 1 aligned actions")
    states = list(latents[:, :context].unbind(1))
    predictions = []
    for offset in range(future_steps):
        last = context - 1 + offset
        begin = max(0, last + 1 - model.config.history_size)
        prediction = model.predict_next(torch.stack(states[begin:], 1), actions[:, begin:last + 1])
        states.append(prediction)
        predictions.append(prediction)
    return torch.stack(predictions, 1)


def prediction_losses(model, latents, actions, future_steps, rollout_weight=1.0):
    context = model.config.history_size
    if latents.shape[1] != context + future_steps:
        raise ValueError("Incorrect target frame count")
    targets = latents.detach()
    teacher = model.predict_sequence(targets[:, :context], actions[:, :context])
    rollout = autoregressive_predictions(model, targets, actions, context, future_steps)
    with torch.autocast(device_type=latents.device.type, enabled=False):
        one_step = F.mse_loss(teacher.float(), targets[:, 1:context + 1].float())
        multistep = F.mse_loss(rollout.float(), targets[:, context:].float())
        cosine = 1 - F.cosine_similarity(rollout.float(), targets[:, context:].float(), dim=-1)
        persistence = 1 - F.cosine_similarity(targets[:, context - 1:context].float(),
                                               targets[:, context:].float(), dim=-1)
        losses = {"loss": one_step + rollout_weight * multistep,
                  "one_step_mse": one_step, "rollout_mse": multistep,
                  "rollout_cosine": cosine.mean(), "persistence_cosine": persistence.mean(),
                  "final_cosine": cosine[:, -1].mean(),
                  "final_persistence_cosine": persistence[:, -1].mean()}
    return losses


class LatentWindows:
    def __init__(self, episodes, context, future_steps):
        self.episodes, self.frames = episodes, context + future_steps
        self.ids = sorted(episodes)
        self.index = [(key, start) for key in self.ids
                      for start in range(len(episodes[key]["latents"]) - self.frames + 1)]

    def __len__(self):
        return len(self.index)

    def batch(self, indices, device):
        selected = [self.index[int(i)] for i in indices]
        latents = torch.stack([self.episodes[key]["latents"][start:start + self.frames]
                               for key, start in selected]).to(device)
        actions = torch.stack([self.episodes[key]["actions"][start:start + self.frames - 1]
                               for key, start in selected]).to(device)
        return latents, actions


@torch.no_grad()
def encode_splits(model, dataset, device, amp, batch_size, progress=None):
    result = {"train": {}, "val": {}}
    banks = {"train": set(), "val": set()}
    with h5py.File(dataset, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError("Dataset must be complete")
        for key, episode in handle["trajectories"].items():
            split = str(episode.attrs["split"])
            if split not in result:
                continue
            actions = torch.from_numpy(np.asarray(episode["action"], dtype=np.float32))
            if len(episode["rgb"]) != len(actions) + 1:
                raise ValueError(f"Misaligned RGB/actions: {key}")
            encoded = []
            for start in range(0, len(episode["rgb"]), batch_size):
                rgb = torch.from_numpy(episode["rgb"][start:start + batch_size]).permute(0, 3, 1, 2).to(device)
                with amp_context(device, amp):
                    encoded.append(model.encode(rgb).float().cpu())
            result[split][key] = {"latents": torch.cat(encoded), "actions": actions}
            banks[split].add(str(episode.attrs["bank_id"]))
            if progress and sum(map(len, result.values())) % 32 == 0:
                progress({"phase": "encoding", "encoded_episodes": sum(map(len, result.values()))})
    if banks["train"] & banks["val"] or set(result["train"]) & set(result["val"]):
        raise ValueError("Train/validation episode or initial-state-bank overlap")
    return result


@torch.no_grad()
def validate(model, windows, indices, batch_size, device, amp, future_steps, rollout_weight):
    totals, count = {}, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(94813)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            latents, actions = windows.batch(batch_indices, device)
            with amp_context(device, amp):
                losses = prediction_losses(model, latents, actions, future_steps, rollout_weight)
            # This is a diagnostic constant: encoder/projector never receive gradients.
            losses["sigreg_frozen_diagnostic"] = model.sigreg(latents.float().transpose(0, 1))
            losses["latent_std"] = latents.float().flatten(0, 1).std(0).mean()
            for key, value in losses.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(f"Nonfinite validation {key}")
                totals[key] = totals.get(key, 0.0) + float(value) * len(batch_indices)
            count += len(batch_indices)
    if not count:
        raise ValueError("Validation has no windows")
    return {key: value / count for key, value in totals.items()} | {"examples": count}


def train(args):
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("FP16 AMP requires CUDA")
    if min(args.max_steps, args.future_steps, args.batch_size, args.val_every, args.encode_batch_size) < 1:
        raise ValueError("Step counts and batch sizes must be positive")
    if args.lr <= 0 or args.rollout_weight <= 0 or args.validation_windows < 0:
        raise ValueError("Invalid learning rate, rollout weight, or validation subset")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    torch.set_num_threads(args.num_threads)
    start = time.time()
    status = {"status": "running", "phase": "loading", "step": 0, "optimizer_updates": 0}

    def update(extra):
        status.update(extra)
        status["elapsed_seconds"] = time.time() - start
        write_json(output / "status.json", status)

    update({})
    try:
        model, original = load_checkpoint(args.checkpoint, device)
        freeze_representation(model)
        original_hash = representation_hash(model)
        encoded = encode_splits(model, args.dataset, device, args.amp, args.encode_batch_size, update)
        train_windows = LatentWindows(encoded["train"], model.config.history_size, args.future_steps)
        val_windows = LatentWindows(encoded["val"], model.config.history_size, args.future_steps)
        if len(train_windows) < args.batch_size or not len(val_windows):
            raise ValueError("Insufficient train/validation windows")
        validation_indices = np.random.default_rng(48193).permutation(len(val_windows))
        if args.validation_windows:
            validation_indices = validation_indices[:args.validation_windows]
        config = {"args": vars(args), "model": model.config_dict(),
                  "source_checkpoint": str(Path(args.checkpoint).resolve()),
                  "source_checkpoint_sha256": file_hash(args.checkpoint),
                  "source_global_step": original.get("global_step"),
                  "dataset": str(Path(args.dataset).resolve()), "dataset_sha256": file_hash(args.dataset),
                  "train_trajectories": train_windows.ids, "val_trajectories": val_windows.ids,
                  "train_windows": len(train_windows), "val_windows": len(val_windows),
                  "validation_indices": validation_indices.tolist(), "validation_shuffle_seed": 48193,
                  "representation_sha256": original_hash, "encoder_frozen": True,
                  "predictor_dropout_and_batchnorm": "eval mode with gradients enabled",
                  "training_inputs": ["cached RGB embeddings", "actions"], "privileged_state": False,
                  "test_split_used": False, "checkpoint_selection_metric": "val_one_step_mse + rollout_weight * val_rollout_mse",
                  "sigreg": "fixed-embedding validation diagnostic only; constant w.r.t. trainable parameters",
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "precision": "fp16" if args.amp else "fp32", "torch_version": str(torch.__version__),
                  "local_git": git_metadata(ROOT), "source_hashes": {
                      str(path.relative_to(ROOT)): file_hash(path) for path in
                      [Path(__file__), ROOT / "latent_stl_lewm/world_model.py", ROOT / "latent_stl_lewm/train.py"]}}
        write_json(output / "config.json", config)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                     weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp, init_scale=512)
        rng = np.random.default_rng(args.seed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        step = updates = 0
        best_val, best_step = float("inf"), 0
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def save(name):
            temp = output / f"{name}.tmp"
            torch.save({"model_config": model.config_dict(), "model_state_dict": model.state_dict(),
                        "config": jsonable(config), "global_step": step, "optimizer_updates": updates,
                        "source_optimizer_updates": original.get("optimizer_updates"),
                        "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                        "best_val": best_val, "best_step": best_step,
                        "representation_sha256": original_hash}, temp)
            temp.replace(output / name)

        def do_validation():
            nonlocal best_val, best_step
            result = validate(model, val_windows, validation_indices, args.batch_size, device,
                              args.amp, args.future_steps, args.rollout_weight)
            if result["loss"] < best_val:
                best_val, best_step = result["loss"], step
                save("best.pt")
            update({"phase": "training", "validation": result, "best_val": best_val, "best_step": best_step})
            return result

        with (output / "metrics.jsonl").open("x", buffering=1) as log:
            initial = do_validation()
            log.write(json.dumps({"step": 0, "validation": initial}) + "\n")
            print(json.dumps({"step": 0, "validation": initial}), flush=True)
            for step in range(1, args.max_steps + 1):
                lr_multiplier = min(1.0, step / max(1, args.warmup_steps))
                lr_multiplier *= 0.1 + 0.9 * (1 + math.cos(math.pi * step / args.max_steps)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * lr_multiplier
                latents, actions = train_windows.batch(rng.integers(len(train_windows), size=args.batch_size), device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, args.amp):
                    losses = prediction_losses(model, latents, actions, args.future_steps, args.rollout_weight)
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                finite = bool(torch.isfinite(grad_norm))
                if not finite and not args.amp:
                    raise FloatingPointError("Nonfinite gradients without AMP")
                scaler.step(optimizer)
                scaler.update()
                updates += int(finite)
                record = {"step": step, "optimizer_updates": updates,
                          "train": {key: float(value.detach()) for key, value in losses.items()},
                          "grad_norm": float(grad_norm) if finite else None,
                          "lr": optimizer.param_groups[0]["lr"], "amp_scale": scaler.get_scale()}
                if step % args.val_every == 0 or step == args.max_steps or stop:
                    record["validation"] = do_validation()
                    save("last.pt")
                    print(json.dumps(record), flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if step % 25 == 0 or step == args.max_steps or stop:
                    update({"step": step, "optimizer_updates": updates, "last_train": record["train"]})
                if stop:
                    break
        if representation_hash(model) != original_hash:
            raise RuntimeError("Frozen representation changed")
        update({"status": "stopped" if stop else "complete", "phase": "complete", "step": step,
                "optimizer_updates": updates, "best_step": best_step, "best_val": best_val,
                "initial_validation": initial, "representation_unchanged": True,
                "best_beats_initial": best_val < initial["loss"],
                "relative_validation_loss_reduction": (initial["loss"] - best_val) / max(initial["loss"], 1e-12),
                "best_checkpoint_sha256": file_hash(output / "best.pt"),
                "last_checkpoint_sha256": file_hash(output / "last.pt"),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None})
    except BaseException as exc:
        update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=0, help="0 means all validation windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
