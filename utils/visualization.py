"""
Visualization utilities for depth map rendering.

Provides colormapped depth images, side-by-side comparisons, and overlay modes.
"""

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm as cm


# Available colormaps for depth visualization
COLORMAPS = {
    "inferno": cm.inferno,
    "magma": cm.magma,
    "viridis": cm.viridis,
    "plasma": cm.plasma,
}


def depth_to_colormap(depth: np.ndarray, colormap: str = "inferno") -> Image.Image:
    """
    Apply a scientific colormap to a normalized depth array.

    Args:
        depth: Normalized depth array (H, W), values in [0, 1]
        colormap: Name of colormap ('inferno', 'magma', 'viridis', 'plasma')

    Returns:
        Colormapped PIL Image (RGB)
    """
    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {list(COLORMAPS.keys())}")

    cmap = COLORMAPS[colormap]
    colored = cmap(depth)  # Returns (H, W, 4) RGBA float array
    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # Drop alpha, convert to uint8

    return Image.fromarray(colored)


def create_side_by_side(
    original: Image.Image,
    depth_colored: Image.Image,
    gap: int = 4,
    bg_color: tuple = (20, 24, 30),
) -> Image.Image:
    """
    Create a side-by-side comparison of original image and depth map.

    Args:
        original: Original PIL Image
        depth_colored: Colormapped depth PIL Image
        gap: Pixel gap between images
        bg_color: Background color for the gap

    Returns:
        Combined PIL Image
    """
    # Resize depth to match original dimensions
    depth_resized = depth_colored.resize(original.size, Image.LANCZOS)

    w, h = original.size
    canvas = Image.new("RGB", (w * 2 + gap, h), bg_color)
    canvas.paste(original, (0, 0))
    canvas.paste(depth_resized, (w + gap, 0))

    return canvas


def create_overlay(
    original: Image.Image,
    depth_colored: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Blend the depth map on top of the original image.

    Args:
        original: Original PIL Image
        depth_colored: Colormapped depth PIL Image
        alpha: Blend factor (0 = only original, 1 = only depth)

    Returns:
        Blended PIL Image
    """
    depth_resized = depth_colored.resize(original.size, Image.LANCZOS)
    original_rgb = original.convert("RGB")

    blended = Image.blend(original_rgb, depth_resized, alpha)
    return blended


def add_depth_legend(
    image: Image.Image,
    colormap: str = "inferno",
    bar_height: int = 24,
    padding: int = 12,
) -> Image.Image:
    """
    Add a color legend bar at the bottom of the image.

    Args:
        image: Input PIL Image
        colormap: Colormap name for the legend
        bar_height: Height of the legend bar
        padding: Padding around the bar

    Returns:
        Image with legend appended at the bottom
    """
    w, h = image.size
    total_height = h + bar_height + padding * 2

    canvas = Image.new("RGB", (w, total_height), (20, 24, 30))
    canvas.paste(image, (0, 0))

    # Create gradient bar
    cmap = COLORMAPS.get(colormap, cm.inferno)
    gradient = np.linspace(0, 1, w - padding * 2).reshape(1, -1)
    gradient = np.repeat(gradient, bar_height, axis=0)
    colored = cmap(gradient)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    bar_img = Image.fromarray(colored)

    canvas.paste(bar_img, (padding, h + padding))

    return canvas
