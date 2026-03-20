---
title: DepthLens
emoji: 🌀
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# DepthLens — Monocular Depth Estimation

Estimate depth from a single image using state-of-the-art deep learning models. Upload any photo and get a detailed depth map showing how far away each part of the scene is.

**[Try the Live Demo →](https://huggingface.co/spaces/getmokshshah/depthlens)**

---

## What It Does

DepthLens takes a single 2D image and predicts a per-pixel depth map — no stereo cameras, no LiDAR, just one photo. The output is a color-mapped visualization showing relative distances: warm colors (red/yellow) for nearby objects and cool colors (blue/purple) for faraway ones.

This is the same core technique used in autonomous vehicles, AR/VR applications, robotics, and 3D scene reconstruction.

## 📁 Project Structure

```
depthlens/
├── app.py                  # Gradio web app (for HuggingFace Spaces)
├── inference.py            # Standalone inference script
├── requirements.txt        # Python dependencies
├── models/
│   └── depth_estimator.py  # Model wrapper with MiDaS integration
├── utils/
│   └── visualization.py    # Depth map coloring and overlays
└── examples/               # Sample images for testing
```

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/getmokshshah/depthlens.git
cd depthlens
pip install -r requirements.txt
```

### 2. Run the Web App Locally

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860` where you can upload images and see depth maps.

### 3. Run Inference from the Command Line

```bash
# Single image
python inference.py --input photo.jpg --output depth_result.png

# Folder of images
python inference.py --input ./photos/ --output ./results/ --batch

# Choose model size
python inference.py --input photo.jpg --output depth.png --model large

# Save raw depth as NumPy array
python inference.py --input photo.jpg --output depth.npy --save-raw
```

## Model Options

| Model | Speed (CPU) | Quality | Memory | Best For |
|-------|-------------|---------|--------|----------|
| `small` (default) | ~0.5s/image | Good | ~200MB | Real-time apps, demos |
| `large` | ~3s/image | Best | ~1GB | High-quality results |

The **small** model (MiDaS v2.1 Small) is optimized for mobile and edge devices. It runs fast on CPU while producing accurate relative depth maps. The **large** model (DPT-Large) uses a Vision Transformer backbone for maximum accuracy.

## Visualization Modes

DepthLens generates multiple visualization styles:

- **Colored Depth Map**: A viridis/inferno/magma colormap applied to the depth prediction, producing a striking false-color image
- **Side-by-Side Comparison**: Original image next to its depth map for easy comparison
- **Depth Overlay**: Semi-transparent depth map blended on top of the original image

## How It Works

1. **Preprocessing**: The input image is resized and normalized to match the model's expected input format using MiDaS transforms
2. **Depth Prediction**: The image passes through a deep neural network (CNN or Vision Transformer) that outputs a per-pixel inverse depth map
3. **Normalization**: Raw depth values are normalized to [0, 1] range for visualization
4. **Colormap Application**: NumPy and Matplotlib apply scientific colormaps to create visually informative depth images

### Architecture Details

The **small** model uses the EfficientNet-Lite backbone with a lightweight decoder, designed for fast inference. The **large** model uses DPT (Dense Prediction Transformer) — a Vision Transformer encoder with convolutional decoder heads that produces sharper depth boundaries and more consistent large-scale depth predictions.

## Understanding the Output

- **Warm colors** (red, orange, yellow) → **close** to the camera
- **Cool colors** (blue, purple) → **far** from the camera
- **Depth values are relative**, not absolute — the model predicts which parts are closer/farther, not exact distances in meters

## Performance

Benchmarked on a 2-core CPU (HuggingFace Spaces free tier):

| Model | Resolution | Inference Time | Peak RAM |
|-------|-----------|----------------|----------|
| Small | 256×256 | ~0.4s | ~350MB |
| Small | 512×512 | ~0.8s | ~500MB |
| Large | 384×384 | ~2.8s | ~1.2GB |

## Configuration

### Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Path to image or folder |
| `--output` | required | Output path for results |
| `--model` | `small` | Model size: `small` or `large` |
| `--colormap` | `inferno` | Colormap: `inferno`, `magma`, `viridis`, `plasma` |
| `--side-by-side` | `False` | Generate side-by-side comparison |
| `--overlay` | `False` | Generate depth overlay on original |
| `--overlay-alpha` | `0.5` | Transparency for overlay mode |
| `--save-raw` | `False` | Save raw depth as .npy file |
| `--batch` | `False` | Process a folder of images |

## Use Cases

- **3D Scene Understanding**: Understand spatial layout from a single photo
- **Autonomous Systems**: Depth perception for robots and drones
- **AR/VR**: Generate depth data for immersive experiences
- **Photography**: Create depth-based focus effects (synthetic bokeh)
- **Accessibility**: Help describe spatial relationships in scenes

## Limitations

- Depth is **relative**, not metric — objects are ranked near-to-far but without real-world distances
- Transparent and reflective surfaces (glass, mirrors, water) can confuse the model
- Very dark or overexposed regions may have unreliable depth predictions
- The model performs best on natural outdoor scenes and indoor rooms

## License

MIT License — free to use for research or commercial projects.

## Credits

- **MiDaS**: Ranftl et al., "Towards Robust Monocular Depth Estimation" (Intel ISL)
- **DPT**: Ranftl et al., "Vision Transformers for Dense Prediction" (ICCV 2021)
- **Built with**: PyTorch, OpenCV, Gradio, NumPy, Matplotlib
