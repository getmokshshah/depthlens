"""
Downloads example images for the DepthLens demo.
Called automatically by app.py on startup if images are missing.
Uses Unsplash Source (free, no API key needed).
"""

import os
import urllib.request

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

EXAMPLE_URLS = {
    "street.jpg": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=640&q=80",
    "landscape.jpg": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=640&q=80",
    "indoor.jpg": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=640&q=80",
}


def download_examples():
    """Download example images if they don't already exist."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    for filename, url in EXAMPLE_URLS.items():
        filepath = os.path.join(EXAMPLES_DIR, filename)
        if os.path.exists(filepath):
            continue
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  Saved to {filepath}")
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")


if __name__ == "__main__":
    download_examples()
