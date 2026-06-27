#!/usr/bin/env python3
"""
Train ResNet50-SAR on collocated SAR patches + ASCAT wind direction labels.

Usage
-----
# Build a synthetic training manifest from mock SAR scenes (dev scaffold):
python -m training.train --build-synthetic --epochs 2

# Train from an existing manifest:
python -m training.train --manifest data/training/manifest.csv --epochs 20

# Resume / fine-tune:
python -m training.train --manifest data/training/manifest.csv --weights ml/weights/resnet_sar_wind.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def build_synthetic_manifest(data_dir: Path, manifest_path: Path, n_scenes: int = 3) -> int:
    """Generate synthetic SAR scenes and collocate with simulated ASCAT."""
    import numpy as np

    from retrieval.provider import MockSentinelProvider
    from training.collocation import build_manifest_from_scene
    from training.dataset import SARWindDataset

    provider = MockSentinelProvider()
    all_samples = []
    bboxes = [
        [76.0, 8.0, 80.0, 13.0],   # Tamil Nadu coast
        [68.0, 20.0, 72.0, 24.0],  # Gujarat / Arabian Sea
        [85.0, 15.0, 90.0, 20.0],  # Bay of Bengal
    ]

    for i in range(n_scenes):
        bbox = bboxes[i % len(bboxes)]
        scene = provider.fetch_data("2024-06-15", bbox)
        scene_time = datetime(2024, 6, 15, 6 + i, 0, 0)
        scene_dir = data_dir / f"scene_{i:03d}"
        samples = build_manifest_from_scene(
            scene["sigma0"],
            scene["inc_angle"],
            bbox,
            scene_time,
            str(scene_dir),
            simulate_ascat=True,
        )
        all_samples.extend(samples)

    return SARWindDataset.write_manifest(all_samples, str(manifest_path), str(data_dir))


def train(
    manifest_path: Path,
    weights_path: Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    resume_weights: str | None = None,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    from ml.resnet_model import build_resnet50_sar
    from training.dataset import SARWindDataset

    dataset = SARWindDataset(str(manifest_path))
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples in {manifest_path}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_sar()
    if model is None:
        raise RuntimeError("PyTorch/torchvision required for training")

    model = model.to(device)
    if resume_weights and Path(resume_weights).exists():
        state = torch.load(resume_weights, map_location=device)
        model.load_state_dict(state)
        logger.info("Resumed from %s", resume_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += inputs.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += inputs.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        logger.info(
            "Epoch %d/%d — train loss %.4f acc %.3f | val loss %.4f acc %.3f",
            epoch, epochs,
            train_loss / max(train_total, 1), train_acc,
            val_loss / max(val_total, 1), val_acc,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            logger.info("Saved checkpoint to %s (val acc %.3f)", weights_path, val_acc)


def main() -> None:
    from config import MODEL_WEIGHTS_PATH, TRAINING_DATA_DIR, TRAINING_MANIFEST

    parser = argparse.ArgumentParser(description="Train ResNet50-SAR wind direction model")
    parser.add_argument("--manifest", type=str, default=str(TRAINING_MANIFEST))
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS_PATH)
    parser.add_argument("--data-dir", type=str, default=str(TRAINING_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--build-synthetic", action="store_true", help="Build dev manifest from mock SAR")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing weights")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data_dir = Path(args.data_dir)
    weights_path = Path(args.weights)

    if args.build_synthetic:
        data_dir.mkdir(parents=True, exist_ok=True)
        n = build_synthetic_manifest(data_dir, manifest_path)
        logger.info("Built synthetic manifest with %d samples", n)

    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}. Run with --build-synthetic first.")

    train(
        manifest_path,
        weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_weights=args.resume,
    )


if __name__ == "__main__":
    main()
