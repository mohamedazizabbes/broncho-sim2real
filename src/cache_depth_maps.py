"""Warm the training image dataset with BREA-Depth maps (run once before training)."""

import argparse
import importlib.util
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import load_data_config, make_dataloader


def load_brea(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load the official BREA-Depth network and return an RGB-image estimator."""
    net_path = repo.expanduser() / "Net.py"
    spec = importlib.util.spec_from_file_location("brea_depth_net", net_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BREA-Depth network from {net_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.AutoEncoder(feature_channels=256)
    state = torch.load(checkpoint.expanduser(), map_location=device, weights_only=True)
    model.load_state_dict(state.get("state_dict", state))
    model.to(device).eval()

    def estimate(rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        image_with_empty_depth = torch.cat((image, torch.zeros_like(image[:, :1])), dim=1)
        with torch.inference_mode():
            # The official inference path returns RGB plus disparity; channel 3 is disparity.
            return model(image_with_empty_depth, real=False)[0, 3].float().cpu().numpy()

    return estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brea-repo", type=Path, required=True, help="Clone of SIRGLab/BREA-Depth")
    parser.add_argument("--brea-checkpoint", type=Path, required=True, help="Depth_MICCAI.pth")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-namespace", default="brea-depth")
    parser.add_argument(
        "--data-config",
        type=Path,
        help="YAML data config; defaults to config/data/data.yaml",
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.brea_checkpoint.expanduser()
    estimator = load_brea(args.brea_repo, checkpoint, device)
    data_config = load_data_config(args.data_config) if args.data_config else load_data_config()

    # Keep inference in this process: CUDA models should not be copied into DataLoader workers.
    loader = make_dataloader(
        sources=data_config.sources,
        image_size=args.image_size or data_config.image_size,
        batch_size=args.batch_size or data_config.batch_size,
        num_workers=0,
        shuffle=False,
        depth_cache_dir=args.cache_dir,
        depth_estimator=estimator,
        depth_cache_namespace=args.cache_namespace or data_config.depth_cache_namespace,
    )
    for _ in tqdm(loader, total=len(loader), desc="Caching BREA depth", unit="batch"):
        pass
    print(f"Cached {len(loader.dataset)} depth maps in {args.cache_dir.expanduser()}")


if __name__ == "__main__":
    main()
