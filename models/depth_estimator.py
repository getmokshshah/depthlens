"""
Depth estimation model wrapper using MiDaS.

Supports two model sizes:
  - small: MiDaS v2.1 Small (EfficientNet-Lite backbone, fast CPU inference)
  - large: DPT-Large (Vision Transformer backbone, highest quality)
"""

import torch
import numpy as np
from PIL import Image


# Model configurations
MODEL_CONFIGS = {
    "small": {
        "repo": "intel-isl/MiDaS",
        "model_name": "MiDaS_small",
        "transform_name": "small_transform",
        "description": "MiDaS v2.1 Small — Fast CPU inference (~0.5s)",
    },
    "large": {
        "repo": "intel-isl/MiDaS",
        "model_name": "DPT_Large",
        "transform_name": "dpt_transform",
        "description": "DPT-Large — Highest quality depth estimation (~3s)",
    },
}


class DepthEstimator:
    """Monocular depth estimation using MiDaS models."""

    def __init__(self, model_size: str = "small", device: str = None):
        """
        Initialize the depth estimator.

        Args:
            model_size: 'small' or 'large'
            device: 'cpu' or 'cuda' (auto-detected if None)
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.config = MODEL_CONFIGS[model_size]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self):
        """Load the MiDaS model and transforms from PyTorch Hub."""
        print(f"Loading {self.config['description']}...")

        # Load model
        self.model = torch.hub.load(
            self.config["repo"],
            self.config["model_name"],
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load(
            self.config["repo"],
            "transforms",
            trust_repo=True,
        )

        if self.model_size == "small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict depth from a PIL Image.

        Args:
            image: Input PIL Image (RGB)

        Returns:
            depth_map: Normalized depth array (H, W) with values in [0, 1].
                       Higher values = closer to camera.
        """
        # Convert PIL to numpy RGB
        img_np = np.array(image.convert("RGB"))

        # Apply MiDaS transform
        input_tensor = self.transform(img_np).to(self.device)

        # Run inference
        prediction = self.model(input_tensor)

        # Resize to original dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        return depth

    def __repr__(self):
        return f"DepthEstimator(model_size='{self.model_size}', device='{self.device}')"
