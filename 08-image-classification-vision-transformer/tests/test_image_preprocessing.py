import numpy as np
from PIL import Image
from src.image_preprocessing import PreprocessingConfig, image_to_numpy


def test_shape_and_dtype():
    image = Image.new("RGB", (32, 20), (255, 0, 0))
    array = image_to_numpy(image, PreprocessingConfig(image_size=(224, 224)))
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32


def test_normalization_range_for_primary_color():
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    array = image_to_numpy(image)
    assert np.isclose(array[0, 0].mean(), 1.0)
    assert np.isclose(array[0, 1].mean(), -1.0)
