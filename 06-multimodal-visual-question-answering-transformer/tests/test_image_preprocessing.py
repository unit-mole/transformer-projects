from io import BytesIO
from PIL import Image
from vqa.image_preprocessing import load_and_validate_image

def test_image_is_converted_to_rgb():
    image = Image.new("RGBA", (32, 24), (255, 0, 0, 100))
    output = load_and_validate_image(image)
    assert output.mode == "RGB"
    assert output.size == (32, 24)

def test_bytes_input():
    image = Image.new("RGB", (10, 10), "blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    output = load_and_validate_image(buffer.getvalue())
    assert output.size == (10, 10)
