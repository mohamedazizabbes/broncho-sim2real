"""Generate RGB images from a depth map with a trained checkpoint.

Examples:
  python src/sample.py depth.npy --checkpoint model.pt --output out/
  python src/sample.py depth.png --checkpoint model.pt --output out/ --samples 4 --solver heun

Inputs:
  .npy            float array, HxW. Raw BREA-style disparity (near = high) by default.
  .png/.tif/.exr  single-channel image, read with its native values.
  --metric        treat the input as CT-derived metric wall distance (larger = farther),
                  e.g. from an AeroPath-style renderer; it is encoded with the frozen
                  continuous BREA profile first.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_encoding import brea_geometry_image
from model_factory import build_model
from utils import conditioning_metadata, prepare_depth_condition, sample_flow


def load_checkpoint(path: Path, device: torch.device, weights: str = "ema"):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("conditioning") != conditioning_metadata():
        raise ValueError("Checkpoint depth conditioning does not match the continuous BREA protocol")
    config = state.get("config") or {}
    architecture = config.get("model")
    if architecture is None:
        raise ValueError("Checkpoint has no saved model configuration")
    model = build_model(architecture)
    model.load_state_dict(state[weights])
    return model.to(device).eval().requires_grad_(False), architecture


def read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth map: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be a 2D array, got shape {depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("depth contains non-finite values")
    return depth


def encode_condition(depth: np.ndarray, size: int, metric: bool, device: torch.device) -> torch.Tensor:
    """Return a 1x1xHxW condition in [-1, 1] using the same encoding as training."""
    if metric:
        encoded = brea_geometry_image(depth, size=size)
        tensor = torch.from_numpy(encoded).float().div(127.5).sub(1)
        return tensor[None, None].to(device)
    resized = cv2.resize(depth, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized)[None, None].to(device)
    return prepare_depth_condition(tensor, "continuous", False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("depth", type=Path, help="depth map (.npy, .png, .tif or .exr)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=1, help="images with independent noise seeds")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--metric", action="store_true", help="input is metric depth (see module docstring)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise SystemExit("--samples and --steps must be positive")
    device = torch.device(args.device)
    model, architecture = load_checkpoint(args.checkpoint, device, args.weights)
    size = args.image_size
    multiple = 2 ** len(architecture.get("channel_mult", [1, 2, 3, 4]))
    if size % multiple:
        raise SystemExit(f"--image-size must be divisible by {multiple}")

    depth = read_depth(args.depth)
    condition = encode_condition(depth, size, args.metric, device)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.depth.stem
    for index in tqdm(range(args.samples), desc="Samples", unit="image"):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        noise = torch.randn((1, 3, size, size), device=device, generator=generator)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            image = sample_flow(model, condition, noise, args.steps, args.solver)
        rgb = ((image[0].float().add(1).mul(127.5).round().clamp(0, 255))
               .byte().permute(1, 2, 0).cpu().numpy())
        target = args.output / f"{stem}-{index:02d}.png"
        if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {target}")
    print(f"Wrote {args.samples} image(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
