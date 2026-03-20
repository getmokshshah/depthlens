"""
DepthLens — Monocular Depth Estimation
Gradio app for HuggingFace Spaces deployment.

Estimates depth from a single image using MiDaS models.
Optimized for free-tier CPU inference.
"""

import time

import gradio as gr
import numpy as np
from PIL import Image

from models import DepthEstimator
from utils import depth_to_colormap, create_side_by_side, create_overlay
from download_examples import download_examples

# ──────────────────────────────────────────────
#  Download example images if missing
# ──────────────────────────────────────────────
download_examples()

# ──────────────────────────────────────────────
#  Load model at startup (small for CPU speed)
# ──────────────────────────────────────────────
print("Starting DepthLens...")
estimator = DepthEstimator(model_size="small")
print("Ready!")


def predict(
    image: Image.Image,
    colormap: str,
    output_mode: str,
    overlay_alpha: float,
) -> tuple:
    """
    Run depth estimation and return results.

    Returns:
        (result_image, depth_colored, stats_string)
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    start = time.time()

    # Run depth estimation
    image_rgb = image.convert("RGB")
    depth = estimator.predict(image_rgb)

    inference_time = time.time() - start

    # Create colormapped depth
    depth_colored = depth_to_colormap(depth, colormap.lower())

    # Create output based on mode
    if output_mode == "Side-by-Side":
        result = create_side_by_side(image_rgb, depth_colored)
    elif output_mode == "Overlay":
        result = create_overlay(image_rgb, depth_colored, alpha=overlay_alpha)
    else:
        result = depth_colored

    # Stats
    w, h = image_rgb.size
    stats = f"{w}×{h} · {inference_time:.2f}s inference · MiDaS Small"

    return result, depth_colored, stats


# ──────────────────────────────────────────────
#  Gradio Interface
# ──────────────────────────────────────────────
with gr.Blocks(
    title="DepthLens — Monocular Depth Estimation",
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
    ),
) as demo:
    gr.Markdown(
        """
        # DepthLens — Monocular Depth Estimation
        Upload any image to estimate per-pixel depth using MiDaS.
        Warm colors = close, cool colors = far.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            colormap = gr.Dropdown(
                choices=["Inferno", "Magma", "Viridis", "Plasma"],
                value="Inferno",
                label="Colormap",
            )
            output_mode = gr.Radio(
                choices=["Depth Map", "Side-by-Side", "Overlay"],
                value="Depth Map",
                label="Output Mode",
            )
            overlay_alpha = gr.Slider(
                minimum=0.2, maximum=0.8, value=0.5, step=0.1,
                label="Overlay Opacity",
                visible=False,
            )
            run_btn = gr.Button("Estimate Depth", variant="primary")
            stats = gr.Textbox(label="Info", interactive=False)

        with gr.Column(scale=1):
            result_image = gr.Image(type="pil", label="Result")
            depth_image = gr.Image(type="pil", label="Depth Map", visible=False)

    # Show/hide overlay slider
    def toggle_overlay(mode):
        return gr.update(visible=(mode == "Overlay"))

    output_mode.change(toggle_overlay, output_mode, overlay_alpha)

    # Run prediction
    run_btn.click(
        fn=predict,
        inputs=[input_image, colormap, output_mode, overlay_alpha],
        outputs=[result_image, depth_image, stats],
    )

    # Examples
    gr.Examples(
        examples=[
            ["examples/street.jpg", "Inferno", "Side-by-Side", 0.5],
            ["examples/landscape.jpg", "Magma", "Depth Map", 0.5],
            ["examples/indoor.jpg", "Viridis", "Overlay", 0.5],
        ],
        inputs=[input_image, colormap, output_mode, overlay_alpha],
        outputs=[result_image, depth_image, stats],
        fn=predict,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
