from PIL import Image
import io
import numpy as np


def read_image_bytes_to_pil(file_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return image


def preprocess_for_model(image: Image.Image, size=(224, 224)) -> np.ndarray:
    """
    Generic preprocessing:
    - resize
    - normalize [0,1]
    - flatten or keep tensor depending on model
    """
    image = image.resize(size)
    arr = np.array(image).astype("float32") / 255.0
    return arr