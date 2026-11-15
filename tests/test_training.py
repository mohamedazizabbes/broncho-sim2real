"""Training bookkeeping: checkpoints, resume guards, cadence and objective wiring."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_factory import build_model  # noqa: E402
from train import epoch_events, flow_matching_losses, restore_training, update_totals  # noqa: E402
from utils import EMA, checkpoint, conditioning_metadata, sample_flow  # noqa: E402


def tiny_model_config():
    return {"architecture": "unet", "base_channels": 16, "channel_mult": [1, 2],
            "blocks_per_level": 1, "attention_head_dim": 8, "detail_channels": 8}


def run_config(epochs=2):
    return OmegaConf.create({
        "model": tiny_model_config(), "data": {}, "learning_rate": 1e-3, "seed": 67,
        "training": {"epochs": epochs, "warmup_epochs": 0, "min_learning_rate": 1e-5,
                     "max_batches_per_epoch": None, "t_sampling": "logit_normal",
                     "repa": {"layer": 1, "weight": 0.0}, "ema_decay": 0.99},
    })


def test_epoch_events_follow_cadence_and_always_checkpoint_the_last_epoch():
    training = OmegaConf.create({"eval_every_epochs": 25, "sample_every_epochs": 5,
                                 "checkpoint_every_epochs": 25, "epochs": 30})
    assert epoch_events(5, training) == (False, True, False)
    assert epoch_events(25, training) == (True, True, True)
    assert epoch_events(30, training) == (False, True, True)


def test_update_totals_accumulates_batch_weighted_sums():
    totals = {}
    update_totals(totals, {"loss": torch.tensor(2.0)}, 4)
    update_totals(totals, {"loss": torch.tensor(1.0)}, 2)
    assert totals["loss"].item() == pytest.approx(10.0)


def test_flow_objective_optimizer_checkpoint_and_resume_round_trip():
    model = build_model(tiny_model_config())
    ema = EMA(model, 0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    image = torch.randn(2, 3, 16, 24)
    depth = torch.randn(2, 1, 16, 24)
    losses = flow_matching_losses(model, None, depth, image, 0.0, t_sampling="logit_normal")
    assert losses["repa_loss"].item() == 0
    losses["loss"].backward()
    assert model.final.weight.grad.abs().sum() > 0
    optimizer.step()
    scheduler.step()
    ema.update(model)
    config = run_config()
    with tempfile.TemporaryDirectory() as directory:
        path = checkpoint(directory, model, ema.model, optimizer, None, scheduler, 1, 1,
                          "last", OmegaConf.to_container(config), scaler=scaler)
        restored = build_model(tiny_model_config())
        restored_ema = EMA(restored, 0.99)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1)
        assert restore_training(path, config, restored, restored_ema, restored_optimizer,
                                None, restored_scheduler, scaler) == (1, 1)
        torch.testing.assert_close(restored.final.weight, model.final.weight)
        torch.testing.assert_close(restored_ema.model.final.weight, ema.model.final.weight)
        best = checkpoint(directory, model, ema.model, optimizer, None, scheduler, 1, 1, "best",
                          OmegaConf.to_container(config))
        assert "optimizer" not in torch.load(best, weights_only=True)
    sampled = sample_flow(restored, depth, image, 2, "heun")
    assert sampled.shape == image.shape and torch.isfinite(sampled).all()


def test_resume_rejects_changed_architecture_and_conditioning():
    model = build_model(tiny_model_config())
    ema = EMA(model, 0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    config = run_config()
    with tempfile.TemporaryDirectory() as directory:
        path = checkpoint(directory, model, ema.model, optimizer, None, scheduler, 1, 1,
                          "last", OmegaConf.to_container(config))
        changed = run_config()
        changed.model.base_channels = 32
        with pytest.raises(ValueError, match="unchanged model"):
            restore_training(path, changed, model, ema, optimizer, None, scheduler,
                             torch.amp.GradScaler("cuda", enabled=False))
        state = torch.load(path, weights_only=True)
        state["conditioning"] = {"protocol": "other"}
        torch.save(state, path)
        assert state["conditioning"] != conditioning_metadata()
        with pytest.raises(ValueError, match="conditioning protocol"):
            restore_training(path, config, model, ema, optimizer, None, scheduler,
                             torch.amp.GradScaler("cuda", enabled=False))
