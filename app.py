"""
DepthLens — Monocular Depth Estimation
Gradio app for HuggingFace Spaces deployment.

Estimates depth from a single image using MiDaS models.
Optimized for free-tier CPU inference.
"""

import os
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


def predict(image, colormap, output_mode, overlay_alpha):
    """Run depth estimation and return result image + stats string."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    start = time.time()

    image_rgb = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
    depth = estimator.predict(image_rgb)

    inference_time = time.time() - start

    depth_colored = depth_to_colormap(depth, colormap.lower())

    if output_mode == "Side-by-Side":
        result = create_side_by_side(image_rgb, depth_colored)
    elif output_mode == "Overlay":
        result = create_overlay(image_rgb, depth_colored, alpha=overlay_alpha)
    else:
        result = depth_colored

    w, h = image_rgb.size
    stats = f"{w}×{h} · {inference_time:.2f}s inference · MiDaS Small"

    return result, stats


# ──────────────────────────────────────────────
#  Build example list (only include files that exist)
# ──────────────────────────────────────────────
example_list = []
for name in ["street.jpg", "landscape.jpg", "indoor.jpg"]:
    path = os.path.join("examples", name)
    if os.path.exists(path):
        example_list.append([path])


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
            )
            run_btn = gr.Button("Estimate Depth", variant="primary")
            stats = gr.Textbox(label="Info", interactive=False)

        with gr.Column(scale=1):
            result_image = gr.Image(type="pil", label="Result")

    run_btn.click(
        fn=predict,
        inputs=[input_image, colormap, output_mode, overlay_alpha],
        outputs=[result_image, stats],
        api_name="predict",
    )

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=[input_image],
        )


if __name__ == "__main__":
    demo.launch()
