"""Independent single-GPU LeWM training and RTX 2080 Ti memory benchmarking."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import TrajectoryWindowDataset, training_action_statistics
from .world_model import LeWMConfig, LeWMWorldModel, ROOT, load_checkpoint


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode()
    return value


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(path):
    def git(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class EpochBatchSampler(Sampler):
    """Deterministic per-epoch shuffling with exact mid-epoch resume."""
    def __init__(self, length, batch_size, seed, epoch, start_batch=0):
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        count = self.length // self.batch_size
        for batch in range(self.start_batch, count):
            yield order[batch * self.batch_size:(batch + 1) * self.batch_size]

    def __len__(self):
        return max(0, self.length // self.batch_size - self.start_batch)


def amp_context(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def move_batch(batch, device):
    if set(batch) != {"rgb", "action"}:
        raise ValueError("Only RGB/action data may enter the image-only world model")
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_step(model, batch, optimizer, scaler, device, amp):
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, amp):
        output = model.training_loss(batch["rgb"], batch["action"])
    if not all(torch.isfinite(v).all().item() for v in output.values()):
        raise FloatingPointError("Nonfinite loss or representation statistic")
    scaler.scale(output["loss"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    finite = bool(torch.isfinite(grad_norm).item())
    if not finite and not amp:
        raise FloatingPointError("Nonfinite gradient without AMP")
    scaler.step(optimizer)  # GradScaler skips an overflowing FP16 update.
    scaler.update()
    result = {k: float(v.detach()) for k, v in output.items()}
    result.update(grad_norm=float(grad_norm) if finite else None,
                  optimizer_update=finite, amp_scale=scaler.get_scale())
    return result


@torch.no_grad()
def validate(model, loader, device, amp, max_batches):
    model.eval()
    totals = {}
    count = 0
    # Validation does not consume the training SIGReg/dropout random stream.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(92817)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch(batch, device)
            with amp_context(device, amp):
                output = model.training_loss(batch["rgb"], batch["action"])
            n = batch["rgb"].shape[0]
            for key, value in output.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite validation {key}")
                totals[key] = totals.get(key, 0.0) + float(value) * n
            count += n
    model.train()
    return {k: v / count for k, v in totals.items()} | {"examples": count}


def model_config(args, action_dim=7):
    return LeWMConfig(action_dim=action_dim, image_size=args.image_size,
                      patch_size=args.patch_size, history_size=args.history_size,
                      backend=args.backend)


def benchmark(args, device):
    """Measure full training forward/backward/AdamW updates, not just inference."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise FileExistsError("Choose a new benchmark output directory")
    results = []
    dataset = TrajectoryWindowDataset(args.dataset, "train", args.history_size, 1) if args.dataset else None
    c = model_config(args, int(dataset.metadata["action_dim"]) if dataset else 7)
    for size in args.benchmark_batches:
        seed_all(args.seed)
        model = optimizer = scaler = batch = None
        try:
            model = LeWMWorldModel(c).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler(device.type, enabled=args.amp, init_scale=1024)
            if dataset:
                from torch.utils.data._utils.collate import default_collate
                if not len(dataset):
                    raise ValueError("Benchmark dataset has no training windows")
                batch = default_collate([dataset[i % len(dataset)] for i in range(size)])
            else:
                batch = {"rgb": torch.rand(size, c.history_size + 1, 3, c.image_size, c.image_size),
                         "action": torch.rand(size, c.history_size, c.action_dim) * 2 - 1}
            batch = move_batch(batch, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            steps = []
            for _ in range(args.benchmark_steps):
                steps.append(train_step(model, batch, optimizer, scaler, device, args.amp))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            result = {"batch_size": size, "status": "ok", "steps": args.benchmark_steps,
                      "seconds_per_step": seconds / args.benchmark_steps,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
                      "optimizer_updates": sum(s["optimizer_update"] for s in steps), "last": steps[-1]}
        except torch.cuda.OutOfMemoryError as exc:
            result = {"batch_size": size, "status": "out_of_memory", "error": str(exc)}
        finally:
            del model, optimizer, scaler, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result), flush=True)
        write_json(output / "benchmark.json", {"model": asdict(c), "precision": "fp16" if args.amp else "fp32",
                                               "data": str(args.dataset) if dataset else "synthetic_memory_test_only",
                                               "device": str(device), "results": results})
    if dataset:
        dataset.close()


def train(args, device):
    if not args.dataset:
        raise ValueError("--dataset is required for training")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and not args.resume:
        raise FileExistsError("Run already exists; choose a new --output or use --resume")
    dataset = TrajectoryWindowDataset(args.dataset, "train", args.history_size, 1)
    validation = TrajectoryWindowDataset(args.dataset, "val", args.history_size, 1)
    if len(dataset) < args.batch_size or not len(validation):
        raise ValueError(f"Need >=batch_size train windows and nonempty val: {len(dataset)} train, {len(validation)} val")
    if set(dataset.trajectory_ids) & set(validation.trajectory_ids):
        raise ValueError("Train/validation trajectory overlap")
    c = model_config(args, int(dataset.metadata["action_dim"]))
    model = LeWMWorldModel(c).to(device)
    if args.backend == "frozen_encoder_learned_predictor" and not args.resume:
        if not args.encoder_checkpoint:
            raise ValueError("Frozen-encoder backend requires a trained --encoder-checkpoint")
        source, source_checkpoint = load_checkpoint(args.encoder_checkpoint)
        if source_checkpoint.get("optimizer_updates", source_checkpoint.get("global_step", 0)) < 1:
            raise ValueError("Frozen-encoder source checkpoint has no training updates")
        model.model.encoder.load_state_dict(source.model.encoder.state_dict(), strict=True)
        model.model.projector.load_state_dict(source.model.projector.state_dict(), strict=True)
        del source, source_checkpoint
    mean, std = training_action_statistics(args.dataset)
    model.action_mean.copy_(torch.from_numpy(mean).to(device))
    model.action_std.copy_(torch.from_numpy(std).to(device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp, init_scale=1024)
    config = {"args": vars(args), "model": asdict(c), "dataset": str(Path(args.dataset).resolve()),
              "dataset_sha256": file_hash(args.dataset), "dataset_metadata": jsonable(dataset.metadata),
              "train_trajectories": dataset.trajectory_ids, "val_trajectories": validation.trajectory_ids,
              "train_windows": len(dataset), "val_windows": len(validation),
              "action_mean": mean.tolist(), "action_std": std.tolist(),
              "action_std_floor": 0.1, "precision": "fp16" if args.amp else "fp32",
              "sigreg_precision": "fp32", "latent_layout": model.latent_layout,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "torch_version": str(torch.__version__),
              "upstream": git_metadata(ROOT / "third_party/le-worldmodel"),
              "local_git": git_metadata(ROOT),
              "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                                [Path(__file__), ROOT / "latent_stl_lewm/world_model.py", ROOT / "latent_stl_lewm/dataset.py"]},
              "encoder_checkpoint_sha256": file_hash(args.encoder_checkpoint) if args.encoder_checkpoint else None,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "training_inputs": ["rgb", "action"], "privileged_state": False,
              "checkpoint_selection_metric": "val_loss (prediction + SIGReg)",
              "scheduler": "linear warmup then cosine over planned optimizer attempts"}
    global_step = optimizer_updates = epoch = next_batch = 0
    best_val = float("inf")
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resumed["model_config"] != asdict(c):
            raise ValueError("Resume model configuration differs")
        if resumed["config"]["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Resume dataset hash differs")
        for key in ("batch_size", "seed", "lr", "weight_decay"):
            if resumed["config"]["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume setting differs: {key}")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["scaler_state_dict"])
        global_step, optimizer_updates = resumed["global_step"], resumed["optimizer_updates"]
        epoch, next_batch, best_val = resumed["epoch"], resumed["next_batch"], resumed["best_val"]
        restore_rng(resumed["rng_state"])
        del resumed
    write_json(output / ("resume_config.json" if args.resume else "config.json"), config)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 992))
    log = open(output / "metrics.jsonl", "a", buffering=1)
    start = time.time()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save(name="last.pt"):
        path = output / name
        temp = path.with_suffix(".pt.tmp")
        torch.save({"format_version": 1, "model_config": asdict(c), "config": jsonable(config),
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "global_step": global_step,
                    "optimizer_updates": optimizer_updates, "epoch": epoch, "next_batch": next_batch,
                    "best_val": best_val, "rng_state": rng_state()}, temp)
        temp.replace(path)

    def emit(values):
        record = {"global_step": global_step, "optimizer_updates": optimizer_updates,
                  "epoch": epoch, "elapsed_seconds": time.time() - start, **values}
        line = json.dumps(record, allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)
        write_json(output / "status.json", {"status": "running", **record})

    total_steps = min(args.max_steps, args.epochs * (len(dataset) // args.batch_size))
    model.train()
    last_validation_step = -1
    consecutive_overflows = 0
    try:
        emit({"event": "start", "planned_steps": total_steps, "parameters": config["parameters"]})
        for epoch in range(epoch, args.epochs):
            sampler = EpochBatchSampler(len(dataset), args.batch_size, args.seed, epoch, next_batch)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(args.seed + epoch + 104729))
            batch_start = next_batch
            for batch_number, batch in enumerate(loader, start=batch_start):
                if global_step >= total_steps or stop_requested:
                    break
                progress = max(0, global_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
                factor = min(1.0, (global_step + 1) / max(1, args.warmup_steps))
                factor *= 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * factor
                metrics = train_step(model, move_batch(batch, device), optimizer, scaler, device, args.amp)
                global_step += 1
                optimizer_updates += int(metrics["optimizer_update"])
                consecutive_overflows = 0 if metrics["optimizer_update"] else consecutive_overflows + 1
                if consecutive_overflows > 10:
                    raise FloatingPointError("More than 10 consecutive FP16 gradient overflows")
                next_batch = batch_number + 1
                if global_step == 1 or global_step % args.log_every == 0:
                    emit({"event": "train", **metrics, "lr": optimizer.param_groups[0]["lr"],
                          "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None})
                if global_step % args.val_every == 0:
                    val = validate(model, val_loader, device, args.amp, args.val_batches)
                    last_validation_step = global_step
                    emit({"event": "validation", **{f"val_{k}": v for k, v in val.items()}})
                    if val["loss"] < best_val:
                        best_val = val["loss"]
                        save("best.pt")
                if global_step % args.checkpoint_every == 0:
                    save()
            if global_step >= total_steps or stop_requested:
                break
            next_batch = 0
        if last_validation_step != global_step:
            val = validate(model, val_loader, device, args.amp, args.val_batches)
            emit({"event": "validation", **{f"val_{k}": v for k, v in val.items()}})
            if val["loss"] < best_val:
                best_val = val["loss"]
                save("best.pt")
        save()
        write_json(output / "status.json", {"status": "interrupted" if stop_requested else "completed",
                                             "global_step": global_step, "optimizer_updates": optimizer_updates,
                                             "epoch": epoch, "best_val_loss": best_val,
                                             "elapsed_seconds": time.time() - start})
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "global_step": global_step,
                                             "error": repr(exc), "elapsed_seconds": time.time() - start})
        raise
    finally:
        log.close()
        dataset.close()
        validation.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=100000)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--resume")
    p.add_argument("--backend", choices=["lewm", "frozen_encoder_learned_predictor"], default="lewm")
    p.add_argument("--encoder-checkpoint")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-batches", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--benchmark-steps", type=int, default=5)
    return p


def main():
    args = parser().parse_args()
    if any(v <= 0 for v in [args.batch_size, args.epochs, args.max_steps, args.log_every,
                            args.val_every, args.val_batches, args.checkpoint_every, args.benchmark_steps]):
        raise ValueError("Batch, step, epoch, and log intervals must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        args.amp = False
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    if args.benchmark:
        benchmark(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
