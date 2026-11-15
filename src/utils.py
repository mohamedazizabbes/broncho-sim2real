import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

CONDITIONING_PROTOCOL = "brea_continuous_near_bright_128_v2"


def conditioning_metadata() -> dict:
    return {"protocol": CONDITIONING_PROTOCOL, "mode": "continuous",
            "native_size": 128, "polarity": "near_bright",
            "percentiles": [0.02, 0.98], "resize": "bicubic_antialias",
            "rgb_resize": "area_down_bicubic_up"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_depth_condition(depth: torch.Tensor, mode: str = "continuous",
                            invert: bool = False) -> torch.Tensor:
    """Normalize raw BREA disparity continuously; output is N1HW in [-1, 1].

    Already encoded uint8 BREA images must use uint8_condition_tensor so their
    calibrated contrast survives unchanged.
    """
    if mode != "continuous":
        raise ValueError("depth conditioning must be continuous (Prototype 2 encoding)")
    size = depth.shape[-2:]
    native = F.interpolate(depth.float(), (128, 128), mode="area")
    limits = torch.quantile(native.flatten(2), native.new_tensor([0.02, 0.98]), dim=-1)
    low, high = limits[0, :, :, None, None], limits[1, :, :, None, None]
    unit = ((native - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
    if invert:
        unit = 1 - unit
    encoded = unit.mul(255).round().div(255)
    resized = F.interpolate(encoded, size, mode="bicubic", align_corners=False, antialias=True)
    return resized.clamp(0, 1).mul(2).sub(1)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        torch._foreach_lerp_(list(self.model.parameters()), list(model.parameters()), 1 - self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)


class REPA(nn.Module):
    """Align intermediate U-Net tokens with frozen DINOv2-B features (training only)."""

    def __init__(self, model_dim: int, layer: int = 1, patch_size: int = 16) -> None:
        super().__init__()
        self.layer = layer
        self.patch_size = patch_size
        self.encoder = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
            .eval()
            .requires_grad_(False)
        )
        self.projector = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 768),
        )

    @torch.no_grad()
    def targets(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image.float().add(1).div(2),
            size=tuple(14 * size // self.patch_size for size in image.shape[-2:]),
            mode="bicubic",
            align_corners=False,
        )
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.encoder.forward_features((image - mean) / std)["x_norm_patchtokens"]

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        predicted = F.normalize(self.projector(features), dim=-1)
        target = self.targets(image).to(predicted.dtype)
        if predicted.shape[1] != target.shape[1]:
            target = self._resize_tokens(target, predicted.shape[1])
        return 1 - (predicted * F.normalize(target, dim=-1)).sum(-1).mean()

    @staticmethod
    def _resize_tokens(tokens: torch.Tensor, count: int) -> torch.Tensor:
        source_size, target_size = int(tokens.shape[1] ** 0.5), int(count ** 0.5)
        if source_size ** 2 != tokens.shape[1] or target_size ** 2 != count:
            raise ValueError("REPA requires square token grids")
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, source_size, source_size)
        tokens = F.interpolate(tokens, (target_size, target_size), mode="bilinear", align_corners=False)
        return tokens.flatten(2).transpose(1, 2)


@torch.inference_mode()
def sample_flow(model: nn.Module, depth: torch.Tensor, initial_noise: torch.Tensor, steps: int,
                solver: str = "euler") -> torch.Tensor:
    """Integrate noise to RGB with Euler or Heun steps; output is clamped to [-1, 1]."""
    if (
        initial_noise.ndim != 4 or depth.ndim != 4
        or initial_noise.shape[0] != depth.shape[0]
        or initial_noise.shape[-2:] != depth.shape[-2:]
    ):
        raise ValueError("initial noise and depth must have matching batch and spatial dimensions")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")

    was_training = model.training
    model.eval()
    image = initial_noise.clone()
    dt = 1 / steps
    try:
        for index in tqdm(range(steps), desc=f"Sampling ({solver})", leave=False):
            t = torch.full((depth.shape[0],), index * dt, device=depth.device)
            velocity = model(image, t, depth).float()
            proposal = image + dt * velocity
            if solver == "heun" and index < steps - 1:
                next_velocity = model(proposal, t + dt, depth).float()
                proposal = image + 0.5 * dt * (velocity + next_velocity)
            image = proposal
    finally:
        model.train(was_training)
    return image.clamp(-1, 1)


def checkpoint(output_dir: str, model: nn.Module, ema_model: nn.Module,
               optimizer: torch.optim.Optimizer, repa: nn.Module,
               scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, step: int,
               name: str | None = None, config: dict | None = None, metrics: dict | None = None,
               scaler: torch.amp.GradScaler | None = None) -> Path:
    """Atomically write last.pt (full training state) or best.pt (inference snapshot)."""
    filename = f"{name}.pt" if name else "last.pt"
    path = Path(output_dir) / "checkpoints" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "ema": ema_model.state_dict(),
        "epoch": epoch, "step": step, "config": config,
        "conditioning": conditioning_metadata(), "metrics": metrics or {},
    }
    # best.pt is an inference snapshot; optimizer state belongs to last.pt.
    if name != "best":
        state.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "repa": repa.projector.state_dict() if repa is not None else None,
                      "scaler": scaler.state_dict() if scaler is not None else {}})
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
