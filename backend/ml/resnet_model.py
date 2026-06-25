"""
ResNet50 architecture for SAR wind direction estimation.

Defines the model class so it can be:
  1. Loaded with trained weights (production use)
  2. Replaced by the deterministic simulator in inference.py (current phase)

Architecture follows:
  "Wind direction retrieval from Sentinel-1 SAR images using ResNet"
  (Shao et al., RSE, 2020, doi:10.1016/j.rse.2020.112056)

Key design choices:
  - Single-channel input (VV polarisation)
  - 36-class softmax output (0–350° in 10° bins)
  - Standard ResNet50 body (5 stages)
  - Compatible with torchvision weights for transfer learning
"""
import logging

logger = logging.getLogger(__name__)

DIR_BINS = 36   # 10° per bin


def build_resnet50_sar():
    """
    Returns a ResNet50 model configured for single-channel SAR wind direction.
    Requires PyTorch + torchvision.

    Returns None (with a warning) if torch is not installed.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=None)   # no ImageNet weights — will be fine-tuned on SAR data

        # Adapt first conv layer: ImageNet uses 3 channels, SAR uses 1
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Adapt final FC layer: ImageNet → 1000 classes, we use 36
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, DIR_BINS)

        logger.info("ResNet50-SAR model built (%d direction bins)", DIR_BINS)
        return model

    except ImportError:
        logger.warning("PyTorch not installed — ResNet model unavailable. Using deterministic simulator.")
        return None


def load_model(weights_path: str = None):
    """
    Build the ResNet50-SAR model and optionally load weights.

    Parameters
    ----------
    weights_path : path to a .pth state-dict file, or None

    Returns
    -------
    Loaded PyTorch model (eval mode) or None if torch unavailable
    """
    model = build_resnet50_sar()
    if model is None:
        return None

    if weights_path:
        try:
            import torch
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            logger.info("Loaded ResNet50-SAR weights from %s", weights_path)
        except Exception as exc:
            logger.warning("Could not load weights from %s: %s", weights_path, exc)

    model.eval()
    return model
