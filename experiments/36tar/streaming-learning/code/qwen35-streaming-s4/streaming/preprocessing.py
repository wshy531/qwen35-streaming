"""Explicit image geometry; bounded canvas with no aspect ratio distortion."""
import numpy as np
from PIL import Image


def fit_canvas(rgb, canvas):
    height, width = canvas
    image = Image.fromarray(rgb)
    scale = min(width / image.width, height / image.height)
    size = (max(1, min(width, round(image.width * scale))),
            max(1, min(height, round(image.height * scale))))
    resized = image.resize(size, Image.Resampling.BICUBIC)
    result = Image.new("RGB", (width, height), (0, 0, 0))
    result.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(result)
