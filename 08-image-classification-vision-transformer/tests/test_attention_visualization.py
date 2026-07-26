import numpy as np
from PIL import Image
from src.attention_visualization import overlay_heatmap


def test_overlay_size():
    image = Image.new("RGB", (64, 32), "white")
    heatmap = np.ones((4, 4), dtype=np.float32)
    output = overlay_heatmap(image, heatmap)
    assert output.size == image.size
