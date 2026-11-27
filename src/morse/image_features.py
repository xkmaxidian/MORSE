"""Six RGB descriptors from H&E patches in loaded image-pixel coordinates."""
import numpy as np


def rgb_features(image, coordinates, patch_size=15):
    """Return [mean_R, mean_G, mean_B, sd_R, sd_G, sd_B] per location.

    Coordinates must already be multiplied by the image scale. An odd
    patch is centered at the nearest integer pixel and edge-padded only
    at the image border. This padding is not a tissue segmentation mask.
    """
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("Patch size must be positive and odd.")
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an RGB-compatible image.")
    image = image[:, :, :3].astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite values.")
    if image.max() > 1.1:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    xy = np.asarray(coordinates, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite image (x,y) coordinates.")
    height, width = image.shape[:2]
    radius = patch_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.empty((len(xy), 6), dtype=np.float32)
    for index, (x, y) in enumerate(xy):
        cx, cy = int(round(float(x))), int(round(float(y)))
        if not (0 <= cx < width and 0 <= cy < height):
            raise ValueError(f"Location {index} lies outside the image; check coordinate scaling.")
        patch = padded[cy:cy + patch_size, cx:cx + patch_size, :].reshape(-1, 3)
        result[index] = np.r_[patch.mean(axis=0), patch.std(axis=0, ddof=0)]
    return result
