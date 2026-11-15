"""Train the depth-conditioned flow U-Net from scratch on one CUDA GPU.

Run with: python src/train.py --config-name unet
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import default_collate
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset import make_train_val_dataloaders
from model_factory import build_model
from utils import (
    EMA,
    REPA,
    checkpoint,
    conditioning_metadata,
    prepare_depth_condition,
    sample_flow,
    seed_everything,
)


def sampling_inputs(loader, count, device, conditioning):
    """Fixed held-out pairs and fixed noise; never pick the best random seed."""
    if count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(count, len(loader.dataset))
    indices = torch.linspace(0, len(loader.dataset) - 1, count).long().tolist()
    depth, image = default_collate([
        loader.dataset[index] for index in tqdm(indices, desc="Load fixed validation pairs")
    ])
    depth = prepare_depth_condition(depth.to(device), conditioning.mode, False)
    image = image.to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(image.shape, device=device, generator=generator)
    return depth, noise, image, indices


def flow_matching_losses(model, repa, depth, image, repa_weight, generator=None, t_sampling="uniform"):
    """Conditional velocity objective; REPA is an optional auxiliary term."""
    if t_sampling == "logit_normal":
        t = torch.randn(image.shape[0], device=image.device, generator=generator).sigmoid()
    elif t_sampling == "uniform":
        t = torch.rand(image.shape[0], device=image.device, generator=generator)
    else:
        raise ValueError("t_sampling must be uniform or logit_normal")
    noise = torch.randn(image.shape, device=image.device, generator=generator)
    noisy_image = t[:, None, None, None] * image + (1 - t[:, None, None, None]) * noise
    if repa_weight > 0:
        if repa is None:
            raise ValueError("Positive REPA weight requires a REPA teacher")
        velocity, features = model(noisy_image, t, depth, feature_layer=repa.layer)
        repa_loss = repa(features, image)
    else:
        velocity = model(noisy_image, t, depth)
        repa_loss = image.new_zeros((), dtype=torch.float32)
    flow_loss = torch.nn.functional.mse_loss(velocity.float(), image.float() - noise.float())
    return {"loss": flow_loss + repa_weight * repa_loss,
            "flow_loss": flow_loss, "repa_loss": repa_loss}


def update_totals(totals, losses, batch_size):
    for name, value in losses.items():
        contribution = value.detach() * batch_size
        if name in totals:
            totals[name].add_(contribution)
        else:
            totals[name] = contribution


def epoch_events(epoch, training):
    """One-based epochs: validation, preview, and checkpoint cadence."""
    return (epoch % training.eval_every_epochs == 0,
            epoch % training.sample_every_epochs == 0,
            epoch % training.checkpoint_every_epochs == 0 or epoch == training.epochs)


def cosine_scheduler(optimizer, warmup_steps, total_steps, min_ratio):
    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.inference_mode()
def validate(model, repa, loader, cfg, device, amp_dtype, use_amp):
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    totals, count = {}, 0
    limit = cfg.training.max_validation_batches or len(loader)
    try:
        for batch_index, (depth, image) in enumerate(tqdm(loader, desc="EMA validation", leave=False)):
            if batch_index >= limit:
                break
            depth = prepare_depth_condition(depth.to(device, non_blocking=True))
            image = image.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses = flow_matching_losses(model, repa, depth, image, cfg.training.repa.weight, generator)
            update_totals(totals, losses, len(image))
            count += len(image)
    finally:
        model.train(was_training)
    return {name: total.item() / count for name, total in totals.items()}


@torch.inference_mode()
def preview(model, depth, noise, reference, cfg, amp_dtype, use_amp):
    """Sample the EMA model and return a condition / target / generated grid."""
    outputs = []
    batch_size = cfg.training.sample_batch_size
    for start in tqdm(range(0, len(depth), batch_size), desc="EMA preview", unit="batch"):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outputs.append(sample_flow(model, depth[start:start + batch_size],
                                       noise[start:start + batch_size],
                                       cfg.training.sample_steps, cfg.training.sample_solver).cpu())
    generated = torch.cat(outputs)
    panels = torch.stack((depth.cpu().expand(-1, 3, -1, -1), reference.cpu(), generated), 1).flatten(0, 1)
    return make_grid(panels, nrow=3, normalize=True, value_range=(-1, 1))


def restore_training(path, cfg, model, ema, optimizer, repa, scheduler, scaler):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("conditioning") != conditioning_metadata():
        raise ValueError("Resume requires the continuous BREA conditioning protocol")
    required = {"model", "ema", "optimizer", "repa", "scheduler", "epoch", "step", "config"}
    if not required.issubset(state):
        raise ValueError("Resume requires a complete last.pt training checkpoint")
    previous = state["config"]
    current = OmegaConf.to_container(cfg, resolve=True)
    for section in ("model", "data"):
        if previous[section] != current[section]:
            raise ValueError(f"Resume requires unchanged {section} configuration")
    for key in ("epochs", "warmup_epochs", "min_learning_rate", "max_batches_per_epoch",
                "t_sampling", "repa", "ema_decay"):
        if previous["training"][key] != current["training"][key]:
            raise ValueError(f"Resume requires unchanged training.{key}")
    if previous["learning_rate"] != current["learning_rate"] or previous["seed"] != current["seed"]:
        raise ValueError("Resume requires unchanged learning rate and split seed")
    model.load_state_dict(state["model"])
    ema.model.load_state_dict(state["ema"])
    if repa is not None:
        repa.projector.load_state_dict(state["repa"])
    elif state["repa"] is not None:
        raise ValueError("Resume without REPA requires a checkpoint without REPA state")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state.get("scaler", {}))
    print(f"Resumed epoch {state['epoch']}, step {state['step']} from {path}", flush=True)
    return state["epoch"], state["step"]


def create_run_directory(cfg):
    """A new run never overwrites an earlier run's checkpoints."""
    root = Path(cfg.training.output_dir).expanduser().resolve()
    run = root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    cfg.training.output_dir = str(run)
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    temporary = root / f".latest-{os.getpid()}"
    temporary.symlink_to(run.name, target_is_directory=True)
    os.replace(temporary, root / "latest")
    print(f"Run directory: {run}", flush=True)
    return run


def run_training(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires one CUDA GPU")
    if cfg.training.depth_conditioning.mode != "continuous":
        raise ValueError("Training requires continuous BREA conditioning")
    for key in ("epochs", "sample_count", "sample_steps", "sample_batch_size", "sample_every_epochs",
                "eval_every_epochs", "checkpoint_every_epochs", "log_every"):
        if cfg.training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    seed_everything(cfg.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    loader, val_loader = make_train_val_dataloaders(
        val_fraction=cfg.data.val_fraction, seed=cfg.seed, augmentation=cfg.data.augmentation,
        source_balanced=cfg.training.source_balanced_sampling,
        sources=cfg.data.sources, image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        depth_cache_dir=cfg.data.depth_cache_dir, depth_cache_namespace=cfg.data.depth_cache_namespace,
        quality_filter=cfg.data.quality_filter,
    )
    output_dir = Path(cfg.training.output_dir)
    manifest = {"conditioning": conditioning_metadata(),
                "quality_filter": getattr(loader.dataset.dataset, "quality_summary", {}),
                "train": [], "validation": []}
    for split, data in tqdm([("train", loader.dataset), ("validation", val_loader.dataset)], desc="Record split"):
        for index in tqdm(data.indices, desc=split, leave=False):
            path, frame = data.dataset.samples[index]
            manifest[split].append({"path": str(path), "frame": frame, "group": data.dataset.groups[index]})
    (output_dir / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with torch.device(device):
        model = build_model(OmegaConf.to_container(cfg.model, resolve=True))
    if cfg.training.get("channels_last", False):
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    repa = (REPA(model.hidden_size, cfg.training.repa.layer, model.patch_size).to(device)
            if cfg.training.repa.weight > 0 else None)
    print(f"Backbone: {type(model).__name__}; parameters: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    ema = EMA(model, cfg.training.ema_decay)
    parameters = list(model.parameters())
    if repa is not None:
        parameters.extend(repa.projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate,
                                  betas=tuple(cfg.training.betas), weight_decay=cfg.training.weight_decay,
                                  fused=True)
    steps_per_epoch = min(len(loader), cfg.training.max_batches_per_epoch or len(loader))
    scheduler = cosine_scheduler(optimizer, cfg.training.warmup_epochs * steps_per_epoch,
                                 cfg.training.epochs * steps_per_epoch,
                                 cfg.training.min_learning_rate / cfg.learning_rate)
    use_amp = cfg.training.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    sample_depth, sample_noise, sample_reference, sample_indices = sampling_inputs(
        val_loader, cfg.training.sample_count, device, cfg.training.depth_conditioning)
    (output_dir / "preview-validation.json").write_text(json.dumps({
        "validation_indices": sample_indices, "noise_seed": 0,
        "steps": cfg.training.sample_steps, "solver": cfg.training.sample_solver,
    }, indent=2) + "\n")
    config = OmegaConf.to_container(cfg, resolve=True)
    log_path = output_dir / "metrics.jsonl"
    step, best_score = 0, float("inf")
    start_epoch = 0
    if cfg.training.resume:
        start_epoch, step = restore_training(cfg.training.resume, cfg, model, ema, optimizer,
                                             repa, scheduler, scaler)
    model.train()
    for epoch in tqdm(range(start_epoch, cfg.training.epochs), desc="Training epochs",
                      initial=start_epoch, total=cfg.training.epochs):
        totals, count = {}, 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch", total=steps_per_epoch)
        for batch_index, (depth, image) in enumerate(progress):
            if batch_index >= steps_per_epoch:
                break
            depth = prepare_depth_condition(depth.to(device, non_blocking=True))
            image = image.to(device, non_blocking=True)
            if cfg.training.get("channels_last", False):
                image = image.contiguous(memory_format=torch.channels_last)
                depth = depth.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses = flow_matching_losses(model, repa, depth, image, cfg.training.repa.weight,
                                              t_sampling=cfg.training.t_sampling)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, cfg.training.grad_clip,
                                           error_if_nonfinite=True, foreach=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            step += 1
            update_totals(totals, losses, len(image))
            count += len(image)
            if step % cfg.training.log_every == 0:
                progress.set_postfix(loss=f"{losses['loss'].item():.4f}")
        evaluation_epoch, preview_epoch, checkpoint_epoch = epoch_events(epoch + 1, cfg.training)
        metrics = {**{f"train/{name}": total.item() / count for name, total in totals.items()},
                   "epoch": epoch + 1, "learning_rate": scheduler.get_last_lr()[0]}
        if evaluation_epoch:
            validation = validate(ema.model, repa, val_loader, cfg, device, amp_dtype, use_amp)
            metrics.update({f"val/ema_{name}": value for name, value in validation.items()})
        if preview_epoch or evaluation_epoch:
            grid = preview(ema.model, sample_depth, sample_noise, sample_reference, cfg, amp_dtype, use_amp)
            preview_dir = output_dir / "samples"
            preview_dir.mkdir(exist_ok=True)
            save_image(grid, preview_dir / f"epoch-{epoch + 1:04d}.png")
        with log_path.open("a") as handle:
            handle.write(json.dumps({"step": step, **metrics}) + "\n")
        progress.write(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        checkpoint_metric = cfg.training.checkpoint_metric
        if checkpoint_metric in metrics:
            score = metrics[checkpoint_metric]
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite {checkpoint_metric}")
            if score < best_score:
                best_score = score
                checkpoint(str(output_dir), model, ema.model, optimizer, repa, scheduler,
                           epoch + 1, step, "best", config, metrics, scaler)
        if checkpoint_epoch:
            checkpoint(str(output_dir), model, ema.model, optimizer, repa, scheduler,
                       epoch + 1, step, "last", config, metrics, scaler)


@hydra.main(version_base=None, config_path="../config", config_name="unet")
def main(cfg: DictConfig):
    if cfg.training.resume:
        cfg.training.resume = str(Path(cfg.training.resume).expanduser().resolve(strict=True))
    create_run_directory(cfg)
    run_training(cfg)


if __name__ == "__main__":
    main()
