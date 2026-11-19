"""CPU-friendly procedural fixture and smoke run for engineering validation only.

This module intentionally does not reproduce or validate the original research quality.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from model_factory import build_model
from utils import EMA, conditioning_metadata, checkpoint, prepare_depth_condition, sample_flow


TINY_MODEL_CONFIG = {
    "architecture": "unet",
    "in_channels": 3,
    "condition_channels": 1,
    "base_channels": 8,
    "channel_mult": [1, 2],
    "blocks_per_level": 1,
    "attention_head_dim": 4,
    "detail_channels": 4,
    "dropout": 0.0,
    "gradient_checkpointing": False,
}


class ProceduralRGBDepth(Dataset):
    """A deterministic procedural paired RGB/depth dataset generated on demand."""

    def __init__(self, size: int = 32, count: int = 8, seed: int = 67) -> None:
        if size <= 0 or size % 4:
            raise ValueError("size must be positive and divisible by 4")
        if count <= 0:
            raise ValueError("count must be positive")
        self.size = size
        self.count = count
        generator = torch.Generator().manual_seed(seed)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        self.y, self.x = y, x
        self.phase = torch.rand(count, generator=generator)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.phase[index] * 6.283185307179586
        radius = torch.sqrt(self.x.square() + self.y.square())
        depth = (torch.cos(3 * radius + float(phase)) + 1) * 0.5
        depth = depth.clamp(0, 1)
        red = depth
        green = (depth * 0.6 + (self.x + 1) * 0.2).clamp(0, 1)
        blue = ((-self.y + 1) * 0.25 + 0.25 * torch.sin(phase)).clamp(0, 1)
        rgb = torch.stack((red, green, blue)).mul(2).sub(1)
        return depth.unsqueeze(0), rgb


def make_dataloader(size: int = 32, count: int = 8, batch_size: int = 2,
                    seed: int = 67) -> DataLoader:
    dataset = ProceduralRGBDepth(size=size, count=count, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_smoke(output: Path, *, device: torch.device, size: int = 32,
                batches: int = 2, steps: int = 2, seed: int = 67) -> Path:
    if batches <= 0 or steps <= 0:
        raise ValueError("batches and steps must be positive")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    loader = make_dataloader(size=size, count=max(batches * 2, 4),
                             batch_size=min(2, max(batches * 2, 4)), seed=seed)
    model = build_model(TINY_MODEL_CONFIG).to(device)
    ema = EMA(model, 0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    model.train()
    iterator = iter(loader)
    for _ in range(batches):
        try:
            depth, image = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            depth, image = next(iterator)
        depth = prepare_depth_condition(depth.to(device), "continuous", False)
        image = image.to(device)
        time = torch.rand(image.shape[0], device=device)
        noise = torch.randn(image.shape, device=device)
        noisy = time[:, None, None, None] * image + (1 - time[:, None, None, None]) * noise
        optimizer.zero_grad(set_to_none=True)
        velocity = model(noisy, time, depth)
        loss = torch.nn.functional.mse_loss(velocity, image - noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("smoke training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

    run = output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "model": TINY_MODEL_CONFIG,
        "data": {"kind": "procedural-rgb-depth", "image_size": size,
                  "count": len(loader.dataset), "seed": seed},
        "learning_rate": 1e-3,
        "seed": seed,
        "training": {"repa": {"weight": 0.0}, "steps": steps,
                     "batches": batches, "engineering_smoke": True},
    }
    checkpoint_path = checkpoint(str(run), model, ema.model, optimizer, None,
                                  scheduler, 1, batches, "smoke", config)

    from sample import load_checkpoint
    reloaded, _ = load_checkpoint(checkpoint_path, device, "ema")
    depth = prepare_depth_condition(torch.full((1, 1, size, size), 0.5,
                                               device=device), "continuous", False)
    noise = torch.randn((1, 3, size, size), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
    with torch.inference_mode():
        sampled = sample_flow(reloaded, depth, noise, steps, "euler")
    if sampled.shape != (1, 3, size, size) or not torch.isfinite(sampled).all():
        raise RuntimeError("reloaded smoke checkpoint produced invalid output")
    print(f"loss={loss.item():.6f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"sample_shape={tuple(sampled.shape)} sample_range=[{sampled.min().item():.3f}, {sampled.max().item():.3f}]")
    return checkpoint_path


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--device", default="cpu" if not torch.cuda.is_available() else "cuda")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=67)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    device = torch.device(args.device)
    train_smoke(args.output, device=device, size=args.size,
                batches=args.batches, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()

