"""Focused engineering smoke tests (not research-quality evaluation)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smoke_reproduction import ProceduralRGBDepth, train_smoke
from sample import load_checkpoint


def test_procedural_pairs_have_shapes_and_ranges():
    dataset = ProceduralRGBDepth(size=32, count=4, seed=67)
    assert len(dataset) == 4
    for index in range(len(dataset)):
        depth, rgb = dataset[index]
        assert depth.shape == (1, 32, 32)
        assert rgb.shape == (3, 32, 32)
        assert depth.min() >= 0 and depth.max() <= 1
        assert rgb.min() >= -1 and rgb.max() <= 1
        assert torch.isfinite(depth).all() and torch.isfinite(rgb).all()


def test_tiny_unet_trains_for_two_batches_and_reloads(tmp_path):
    checkpoint = train_smoke(tmp_path / "smoke", device=torch.device("cpu"),
                             size=32, batches=2, steps=2, seed=67)
    assert checkpoint.is_file()
    assert checkpoint.name == "smoke.pt"
    reloaded, config = load_checkpoint(checkpoint, torch.device("cpu"), "ema")
    assert config["base_channels"] == 8
    depth = torch.full((1, 1, 32, 32), 0.5)
    noise = torch.randn(1, 3, 32, 32, generator=torch.manual_seed(67))
    from utils import prepare_depth_condition, sample_flow
    condition = prepare_depth_condition(depth, "continuous", False)
    with torch.inference_mode():
        sampled = sample_flow(reloaded, condition, noise, 2, "euler")
    assert sampled.shape == noise.shape
    assert sampled.min() >= -1 and sampled.max() <= 1
    assert torch.isfinite(sampled).all()
