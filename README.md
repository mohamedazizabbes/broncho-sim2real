# Depth-Conditioned Bronchoscopy Image Generation

*A UNet flow-matching model that turns a depth map into a bronchoscopy-style RGB image.*

## ⚡ Highlights

- **Depth-conditioned generation:** the model maps a depth map plus noise to an RGB image, trained with a conditional flow-matching objective.
- **Pixel-space U-Net:** a 100M model that runs very fast even on CPUs, deployed in INT8 precision

![`assets/airway-unet-demo.mp4`](assets/airway-unet-demo.mp4)

## 🛠️ Setup

Install dependencies (Python 3.10+, PyTorch 2.1+, a CUDA GPU for training):

```bash
git clone https://github.com/stefanasandei/broncho-sim2real.git
cd broncho-sim2real
pip install -r requirements.txt
```

Sampling runs on CPU but is slow; training requires one CUDA GPU.

### Model checkpoint

The released checkpoint is a fine-tuned U-Net with EMA weights. It is not yet published in this repository; place a downloaded `best.pt` anywhere and pass its path to `src/sample.py`. The checkpoint embeds its architecture and the conditioning protocol, and the loader refuses checkpoints that do not match the current encoding.

## 🚀 Quickstart

### 1. Inference

Generate RGB images from a depth map. Accepted inputs are `.npy` (float HxW), or a single-channel `.png`, `.tif` or `.exr`:

```bash
python src/sample.py path/to/depth.npy --checkpoint best.pt --output outputs/ --samples 4
```

By default the depth is treated as raw BREA-style disparity (near = high). For CT-derived metric wall distance (larger = farther), for example depth rendered from AeroPath-style airway segmentations, add `--metric`; it is encoded with the frozen profile in `src/depth_encoding.py` before conditioning.

Useful flags: `--steps 100 --solver heun` (default), `--solver euler`, `--weights model` for the non-EMA weights, `--image-size 256`.

### 2. Training

Train the U-Net from scratch on one CUDA GPU:

```bash
python src/train.py --config-name unet
```

Settings live in [`config/unet.yaml`](config/unet.yaml). Override any key on the command line, for example:

```bash
python src/train.py --config-name unet training.repa.weight=0.0 training.epochs=20
```

Each run writes to a new timestamped directory under `runs/unet/`, with a `latest` symlink. It saves `checkpoints/best.pt` (best EMA validation flow loss), `checkpoints/last.pt` (full state for resuming), preview grids in `samples/`, and `metrics.jsonl`. Resume with:

```bash
python src/train.py --config-name unet training.resume=runs/unet/latest/checkpoints/last.pt
```

The auxiliary REPA term (`training.repa.weight`, default 0.5) uses frozen DINOv2 features loaded through `torch.hub` on first use. Set it to `0.0` to train without a network download.

### 3. Data preparation

Training needs RGB image sources and a directory of cached BREA-Depth maps.

1. Edit [`config/data/data.yaml`](config/data/data.yaml) so `sources` points at your images or videos. The defaults are placeholders.
2. Cache BREA-Depth maps once, using a local clone of the BREA-Depth code and its checkpoint:

   ```bash
   python src/cache_depth_maps.py \
     --brea-repo /path/to/BREA-Depth \
     --brea-checkpoint /path/to/Depth_MICCAI.pth \
     --cache-dir data/brea-depth
   ```

3. Train as above. Images pass a sharpness, contrast and colour quality filter before the group-based train/validation split.

Dataset files are not included.

## Limitations

This is research code. The model has not been evaluated for clinical use and must not be used for diagnosis or treatment decisions. This repository reports no benchmark numbers; validation loss and preview grids are the only training-time signals it produces.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
