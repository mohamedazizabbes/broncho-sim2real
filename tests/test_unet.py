"""U-Net contracts: configuration, conditioning, REPA features, checkpoints and sampling."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_factory import build_model  # noqa: E402
from train import flow_matching_losses  # noqa: E402
from unet import FlowUNet  # noqa: E402
from utils import sample_flow  # noqa: E402


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {"architecture": "unet", "base_channels": 16, "channel_mult": [1, 2],
            "blocks_per_level": 1, "attention_head_dim": 8, "detail_channels": 8}


def active_model():
    model = build_model(tiny_config())
    # The fresh head is zero-initialized; activate every branch for the checks below.
    for module in tqdm(list(model.modules()), desc="Activate test branches", leave=False):
        if isinstance(module, torch.nn.Conv2d) and torch.count_nonzero(module.weight) == 0:
            torch.nn.init.normal_(module.weight, std=0.02)
    return model


def inputs(device="cpu"):
    return (torch.randn(2, 3, 16, 24, device=device),
            torch.tensor([0.2, 0.8], device=device),
            torch.randn(2, 1, 16, 24, device=device))


def test_config_composition_and_default_architecture():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="unet")
    assert cfg.model.architecture == "unet"
    assert cfg.training.depth_conditioning.mode == "continuous"
    model = build_model(cfg.model)
    assert isinstance(model, FlowUNet)
    assert 90_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 110_000_000
    with pytest.raises(ValueError, match="Unknown flow architecture"):
        build_model({"architecture": "typo"})


def test_spatial_condition_time_and_repa_feature_contract():
    model = active_model().eval()
    x, t, depth = inputs()
    with torch.no_grad():
        output, features = model(x, t, depth, feature_layer=1)
        changed_depth = model(x, t, depth.flip(-1))
        changed_time = model(x, 1 - t, depth)
    assert output.shape == x.shape
    assert features.shape == (2, (16 // model.patch_size) * (24 // model.patch_size), model.hidden_size)
    assert (output - changed_depth).abs().max() > 1e-5
    assert (output - changed_time).abs().max() > 1e-5
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="divisible"):
        model(x[:, :, :-1], t, depth[:, :, :-1])
    with pytest.raises(ValueError, match="depth must match"):
        model(x, t, depth[:1])
    with pytest.raises(ValueError, match="one timestep"):
        model(x, t[:1], depth)
    with pytest.raises(ValueError, match="feature_layer=1"):
        model(x, t, depth, feature_layer=8)


def test_checkpointed_backward_matches_regular_and_channels_last():
    model = active_model().train()
    recomputed = copy.deepcopy(model).to(memory_format=torch.channels_last)
    recomputed.gradient_checkpointing = True
    x, t, depth = inputs()
    first = model(x, t, depth)
    second = recomputed(x.contiguous(memory_format=torch.channels_last), t, depth)
    torch.testing.assert_close(first, second, atol=2e-6, rtol=1e-4)
    first.square().mean().backward()
    second.square().mean().backward()
    for (name, left), right in tqdm(list(zip(model.named_parameters(), recomputed.parameters())),
                                    desc="Check gradients"):
        assert left.grad is not None, name
        assert torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=1e-4)


def test_optional_repa_backpropagates_through_bottleneck():
    class Teacher(torch.nn.Module):
        layer = 1

        def forward(self, features, image):
            return features.square().mean()

    model = active_model()
    image, _, depth = inputs()
    losses = flow_matching_losses(model, Teacher(), depth, image, 0.5)
    losses["loss"].backward()
    assert model.middle2.conv2.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires a REPA teacher"):
        flow_matching_losses(model, None, depth, image, 0.5)


def test_euler_and_heun_samplers_are_finite_and_clamped():
    model = active_model().eval()
    x, _, depth = inputs()
    for solver in ("euler", "heun"):
        sampled = sample_flow(model, depth, x, 3, solver)
        assert sampled.shape == x.shape
        assert sampled.min() >= -1 and sampled.max() <= 1
        assert torch.isfinite(sampled).all()
    with pytest.raises(ValueError, match="solver"):
        sample_flow(model, depth, x, 2, "rk4")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_forward_matches_fp32_within_tolerance():
    model = active_model().cuda().eval()
    image, t, depth = inputs("cuda")
    with torch.inference_mode():
        reference = model(image, t, depth)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low_precision = model(image, t, depth)
    torch.testing.assert_close(low_precision.float(), reference, atol=5e-2, rtol=5e-2)
