"""Build the flow U-Net from a config dictionary or a checkpoint's saved config."""

from unet import FlowUNet


def build_model(config):
    architecture = dict(config)
    name = architecture.pop("architecture", "unet")
    if name == "unet":
        return FlowUNet(**architecture)
    raise ValueError(f"Unknown flow architecture: {name!r}; expected unet")
