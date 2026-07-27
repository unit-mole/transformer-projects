from pathlib import Path
from PIL import Image
from src.image_preprocessing import load_image_rgb, save_exif_free, validate_dimensions


def test_load_and_save_rgb(tmp_path: Path):
    source = tmp_path / "input.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(source)
    image = load_image_rgb(source)
    assert image.mode == "RGB"
    validate_dimensions(image)
    target = save_exif_free(image, tmp_path / "output.png")
    assert target.exists()
