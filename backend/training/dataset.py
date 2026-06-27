"""
PyTorch Dataset for SAR wind-direction training.

Expects a manifest CSV with columns:
  patch_path, label_bin, wind_dir, wind_speed, lat, lon
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def patch_to_tensor(patch: np.ndarray):
    """Convert a SAR patch to a normalised PyTorch tensor (1, H, W)."""
    import torch

    arr = patch.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr)) or 1.0
        arr = np.nan_to_num(arr, nan=mean)
        arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


class SARWindDataset:
    """Dataset over a training manifest CSV."""

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples: List[dict] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            logger.warning("Manifest not found: %s", self.manifest_path)
            return

        with self.manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        logger.info("Loaded %d training samples from %s", len(self.samples), self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        import torch

        row = self.samples[idx]
        patch_path = Path(row["patch_path"])
        patch = np.load(patch_path)
        tensor = patch_to_tensor(patch)
        if self.transform:
            tensor = self.transform(tensor)

        label = int(row["label_bin"])
        return tensor, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def write_manifest(samples: List, output_path: str, data_dir: str) -> int:
        """Write collocated samples to manifest CSV."""
        from training.collocation import direction_to_bin

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data_root = Path(data_dir)

        rows = []
        for sample in samples:
            patch_path = data_root / sample.patch_path
            rows.append({
                "patch_path": str(patch_path),
                "label_bin": direction_to_bin(sample.wind_dir),
                "wind_dir": f"{sample.wind_dir:.2f}",
                "wind_speed": f"{sample.wind_speed:.2f}",
                "lat": f"{sample.lat:.5f}",
                "lon": f"{sample.lon:.5f}",
            })

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["patch_path", "label_bin", "wind_dir", "wind_speed", "lat", "lon"],
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Wrote %d rows to %s", len(rows), out)
        return len(rows)
