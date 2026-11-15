"""Image dataloader with optional cached BREA-Depth maps and a quality filter."""

import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2
from tqdm import tqdm

CONFIG_PATH = Path(__file__).parents[1] / "config" / "data" / "data.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


DepthEstimator = Callable[[np.ndarray], np.ndarray]
Sample = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class DepthImageDataset(Dataset[Sample]):
    """Return ``(depth, image)`` pairs when depth caching is enabled."""

    def __init__(
        self,
        sources: Sequence[Mapping[str, Any]],
        image_size: int = 256,
        depth_cache_dir: str | Path | None = None,
        depth_estimator: DepthEstimator | None = None,
        depth_cache_namespace: str = "brea-depth",
        augmentation: Mapping[str, float] | None = None,
        quality_filter: Mapping[str, float] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.samples: list[tuple[Path, int | None]] = []
        self.groups: list[str] = []
        self._capture: cv2.VideoCapture | None = None
        self._open_video: Path | None = None
        for source in sources:
            self._add_source(source)
        if not self.samples:
            raise RuntimeError("No images were found")

        self.image_size = image_size
        self.quality_filter = dict(quality_filter) if quality_filter else None
        if self.quality_filter and self.quality_filter.get("enabled", True):
            self._filter_low_quality_samples()
        self.depth_cache_dir = (
            Path(depth_cache_dir).expanduser() if depth_cache_dir is not None else None
        )
        self.depth_estimator = depth_estimator
        self.depth_cache_namespace = depth_cache_namespace
        self.augmentation = dict(augmentation) if augmentation else None
        if self.depth_cache_dir is not None:
            self.depth_cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def _filter_low_quality_samples(self) -> None:
        """Remove target frames that teach blur, blue casts, or scope borders."""
        kept_samples: list[tuple[Path, int | None]] = []
        kept_groups: list[str] = []
        rejected: Counter[str] = Counter()
        candidates = Counter(group.split(":", 1)[0] for group in self.groups)
        for sample, group in tqdm(
            zip(self.samples, self.groups, strict=True), total=len(self.samples),
            desc="Filtering RGB targets", unit="image",
        ):
            path, frame_index = sample
            image = (
                self._read_image(path)
                if frame_index is None
                else self._read_video_frame(path, frame_index)
            )
            reason = self._quality_rejection_reason(image)
            if reason is None:
                kept_samples.append(sample)
                kept_groups.append(group)
            else:
                rejected[f"{group.split(':', 1)[0]}/{reason}"] += 1
        self._close_video()
        if not kept_samples:
            raise RuntimeError("Image quality filter rejected every sample")
        self.samples, self.groups = kept_samples, kept_groups
        retained = Counter(group.split(":", 1)[0] for group in kept_groups)
        self.quality_summary = {
            "candidates": dict(candidates), "retained": dict(retained),
            "rejected": dict(rejected),
        }
        details = ", ".join(f"{key}={value}" for key, value in sorted(rejected.items()))
        print(
            f"Quality filter retained {len(kept_samples)} samples and rejected "
            f"{sum(rejected.values())}" + (f" ({details})" if details else "")
        )
        print(f"Retained by source: {dict(retained)}")

    def _resize_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Use identical resizing for quality checks and actual RGB targets."""
        interpolation = (
            cv2.INTER_AREA if min(bgr.shape[:2]) >= self.image_size else cv2.INTER_CUBIC
        )
        return cv2.resize(bgr, (self.image_size, self.image_size), interpolation=interpolation)

    def _quality_rejection_reason(self, bgr: np.ndarray) -> str | None:
        config = self.quality_filter
        if config is None:
            return None
        image = self._resize_bgr(bgr)
        rgb_mean = image[:, :, ::-1].astype(np.float32).mean(axis=(0, 1))
        if rgb_mean[2] / max(float(rgb_mean[0]), 1.0) > float(
            config["max_blue_red_ratio"]
        ):
            return "blue"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        border_width = max(
            1, round(self.image_size * float(config["border_width_fraction"]))
        )
        border = np.concatenate(
            (
                gray[:border_width].ravel(),
                gray[-border_width:].ravel(),
                gray[:, :border_width].ravel(),
                gray[:, -border_width:].ravel(),
            )
        )
        black_fraction = np.mean(border <= int(config["black_threshold"]))
        if black_fraction > float(config["max_black_border_fraction"]):
            return "border"
        # Measure tissue, excluding scope boundaries, peripheral overlays, and
        # clipping edges. Whole-image Laplacian variance rewards hard borders.
        mask = gray > int(config["black_threshold"])
        mask &= gray < int(config.get("highlight_threshold", 245))
        erosion = max(1, round(self.image_size * float(config.get("tissue_erosion_fraction", 0.015))))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * erosion + 1,) * 2, np.uint8)).astype(bool)
        margin = max(1, round(self.image_size * float(config.get("tissue_margin_fraction", 0.1))))
        mask[:margin] = mask[-margin:] = False
        mask[:, :margin] = mask[:, -margin:] = False
        if mask.mean() < float(config.get("min_tissue_fraction", 0.2)):
            return "insufficient_tissue"
        if float(gray[mask].std()) < float(config["min_intensity_std"]):
            return "low_contrast"
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)[mask]
        # Prevent a few residual text/edge pixels from dominating the score.
        limit = np.quantile(np.abs(laplacian), float(config.get("sharpness_clip_quantile", 0.99)))
        sharpness = float(np.clip(laplacian, -limit, limit).var())
        if sharpness < float(config["min_laplacian_variance"]):
            return "blur"
        return None

    def _add_source(self, source: Mapping[str, Any]) -> None:
        root = Path(str(source["path"])).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        paths = sorted(
            path
            for path in root.glob(str(source.get("pattern", "**/*")))
            if self._is_data_file(path)
        )
        source_type = str(source["type"])
        if source_type == "images":
            paths = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
            group_size = int(source.get("split_group_size", 1))
            group_parent = int(source.get("split_group_parent", 1))
            if group_size <= 0 or group_parent <= 0:
                raise ValueError("split grouping values must be positive")
            self.samples.extend((path, None) for path in paths)
            self.groups.extend(
                f"{source['name']}:{index // group_size}"
                if group_size > 1
                else f"{source['name']}:{path.parents[group_parent - 1]}"
                for index, path in enumerate(paths)
            )
        elif source_type == "videos":
            stride = int(source.get("frame_stride", 1))
            if stride <= 0:
                raise ValueError("frame_stride must be positive")
            for path in paths:
                frames = range(0, self._frame_count(path), stride)
                self.samples.extend((path, index) for index in frames)
                self.groups.extend(f"{source['name']}:{path}" for _ in frames)
        else:
            raise ValueError(f"Unknown source type: {source_type!r}")

    @staticmethod
    def _is_data_file(path: Path) -> bool:
        return path.is_file() and not any(
            part.startswith(".") or part == "__MACOSX" for part in path.parts
        )

    @staticmethod
    def _frame_count(path: Path) -> int:
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count <= 0:
            raise RuntimeError(f"Could not read video: {path}")
        return count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        path, frame_index = self.samples[index]
        image = (
            self._read_image(path)
            if frame_index is None
            else self._read_video_frame(path, frame_index)
        )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.image_transform(self._resize_bgr(rgb))
        if self.depth_cache_dir is None:
            return self._normalize_image(self._augment_image(image_tensor))

        depth = self._load_or_compute_depth(path, frame_index, rgb)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        depth_tensor, image_tensor = self._augment_pair(depth_tensor, image_tensor)
        return depth_tensor, self._normalize_image(image_tensor)

    @staticmethod
    def _normalize_image(image: torch.Tensor) -> torch.Tensor:
        return image.sub(0.5).div(0.5)

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augmentation:
            return image
        image = v2.ColorJitter(
            brightness=self.augmentation["color_jitter"],
            contrast=self.augmentation["color_jitter"],
            saturation=self.augmentation["color_jitter"],
            hue=self.augmentation["hue_jitter"],
        )(image)
        gamma = 1 + self.augmentation["gamma"] * (2 * torch.rand(()) - 1)
        return v2.functional.adjust_gamma(image, float(gamma))

    def _augment_pair(
        self, depth: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augmentation:
            return depth, image
        if torch.rand(()) < self.augmentation["horizontal_flip"]:
            depth, image = depth.flip(-1), image.flip(-1)
        if torch.rand(()) < self.augmentation["vertical_flip"]:
            depth, image = depth.flip(-2), image.flip(-2)
        if torch.rand(()) < self.augmentation["quarter_turn_probability"]:
            turns = int(torch.randint(1, 4, ()).item())
            image = torch.rot90(image, turns, (-2, -1))
            depth = torch.rot90(depth, turns, (-2, -1))
        return depth, self._augment_image(image)

    def _load_or_compute_depth(
        self, path: Path, frame_index: int | None, rgb: np.ndarray
    ) -> np.ndarray:
        cache_path = self.depth_cache_path(path, frame_index)
        if cache_path.is_file():
            depth = np.load(cache_path, allow_pickle=False)
            if not np.isfinite(depth).all():
                raise ValueError(f"Cached depth map {cache_path} has non-finite values")
        else:
            if self.depth_estimator is None:
                raise FileNotFoundError(
                    f"Depth cache miss for {path}"
                    "; run src/cache_depth_maps.py before cache-only training"
                )
            depth = np.asarray(self.depth_estimator(rgb), dtype=np.float32).squeeze()
            if depth.ndim != 2:
                raise ValueError(
                    f"Depth estimator returned shape {depth.shape}, expected HxW"
                )
            depth = cv2.resize(
                depth,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32, copy=False)
            if not np.isfinite(depth).all():
                raise ValueError(
                    f"Depth estimator returned non-finite values for {path}"
                )
            self._atomic_save(cache_path, depth)

        expected = (self.image_size, self.image_size)
        if depth.shape != expected:
            raise ValueError(
                f"Cached depth map {cache_path} has shape {depth.shape}, expected {expected}"
            )
        return np.ascontiguousarray(depth, dtype=np.float32)

    def depth_cache_path(self, path: Path, frame_index: int | None) -> Path:
        """Return a deterministic cache path that changes when the source changes."""
        if self.depth_cache_dir is None:
            raise RuntimeError("Depth caching is not enabled")
        stat = path.stat()
        identity = (
            f"{path.resolve()}\0{frame_index}\0{stat.st_size}\0{stat.st_mtime_ns}"
            f"\0{self.image_size}\0{self.depth_cache_namespace}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.depth_cache_dir / digest[:2] / f"{digest}.npy"

    @staticmethod
    def _atomic_save(path: Path, depth: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                np.save(temporary_file, depth, allow_pickle=False)
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _read_image(path: Path) -> Any:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        return image

    def _read_video_frame(self, path: Path, frame_index: int) -> Any:
        if path != self._open_video or self._capture is None:
            self._close_video()
            self._capture = cv2.VideoCapture(str(path))
            self._open_video = path
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame

    def _close_video(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._open_video = None

    def __del__(self) -> None:
        self._close_video()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local OpenCV state when spawning data workers."""
        state = self.__dict__.copy()
        state["_capture"] = None
        state["_open_video"] = None
        return state


def load_data_config(path: str | Path = CONFIG_PATH) -> Any:
    return OmegaConf.load(path)


def make_dataloader(
    sources: Sequence[Mapping[str, Any]] | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool = True,
    depth_cache_dir: str | Path | None = None,
    depth_estimator: DepthEstimator | None = None,
    depth_cache_namespace: str | None = None,
    augmentation: Mapping[str, float] | None = None,
    quality_filter: Mapping[str, float] | None = None,
) -> DataLoader[Sample]:
    config = load_data_config()
    dataset = DepthImageDataset(
        sources=sources if sources is not None else config.sources,
        image_size=image_size if image_size is not None else config.image_size,
        depth_cache_dir=depth_cache_dir
        if depth_cache_dir is not None
        else config.get("depth_cache_dir"),
        depth_estimator=depth_estimator,
        depth_cache_namespace=(
            depth_cache_namespace
            if depth_cache_namespace is not None
            else config.get("depth_cache_namespace", "brea-depth")
        ),
        augmentation=augmentation,
        quality_filter=quality_filter,
    )
    workers = num_workers if num_workers is not None else config.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_train_val_dataloaders(
    *,
    val_fraction: float,
    seed: int,
    augmentation: Mapping[str, float],
    source_balanced: bool = False,
    **kwargs: Any,
) -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    """Create a deterministic group split with augmentation only on training data."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    train_loader = make_dataloader(augmentation=augmentation, **kwargs)
    train_dataset = train_loader.dataset
    val_dataset = DepthImageDataset.__new__(DepthImageDataset)
    val_dataset.__dict__ = train_dataset.__dict__.copy()
    val_dataset.augmentation = None
    val_dataset._capture = None
    val_dataset._open_video = None

    group_counts = Counter(train_dataset.groups)
    source_groups: dict[str, list[str]] = defaultdict(list)
    for group in group_counts:
        source_groups[group.split(":", 1)[0]].append(group)
    val_groups: set[str] = set()
    for groups in source_groups.values():
        groups.sort(
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest()
        )
        target = sum(group_counts[group] for group in groups) * val_fraction
        selected = 0
        for group in groups:
            if selected and abs(selected - target) <= abs(
                selected + group_counts[group] - target
            ):
                break
            val_groups.add(group)
            selected += group_counts[group]
    train_indices = [
        index
        for index, group in enumerate(train_dataset.groups)
        if group not in val_groups
    ]
    val_indices = [
        index for index, group in enumerate(train_dataset.groups) if group in val_groups
    ]
    if not train_indices or not val_indices:
        raise RuntimeError("train/validation split produced an empty partition")

    loader_options = {
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "persistent_workers": train_loader.num_workers > 0,
    }
    train_subset = Subset(train_dataset, train_indices)
    sampler = None
    if source_balanced:
        source_counts = Counter(
            train_dataset.groups[index].split(":", 1)[0] for index in train_indices
        )
        weights = [
            1.0 / source_counts[train_dataset.groups[index].split(":", 1)[0]]
            for index in train_indices
        ]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return (
        DataLoader(
            train_subset,
            shuffle=sampler is None,
            sampler=sampler,
            **loader_options,
        ),
        DataLoader(Subset(val_dataset, val_indices), shuffle=False, **loader_options),
    )


if __name__ == "__main__":
    loader = make_dataloader(num_workers=0)
    depths, images = next(iter(loader))
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"First depth batch shape: {tuple(depths.shape)}")
    print(f"First image batch shape: {tuple(images.shape)}")
