import numpy as np

from .base import PreprocessingStep


class IdentityStep(PreprocessingStep):
    """Pass-through — no transformation. Used for mode='rgb'."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image

    @property
    def name(self) -> str:
        return "identity"


class GreenChannelStep(PreprocessingStep):
    """Extract the green channel from an RGB image.

    The green channel provides the highest vessel-background contrast in
    retinal fundus images (Fraz et al. 2012; Guo et al. 2021).

    Input : (H, W, 3) uint8
    Output: (H, W, 1) uint8
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        green = image[:, :, 1]
        return np.expand_dims(green, axis=-1)

    @property
    def name(self) -> str:
        return "green_channel"


class RGBCLAHEStep(PreprocessingStep):
    """Apply CLAHE independently to each RGB channel, preserving 3-channel output.

    Applies CLAHE to the R, G, and B channels separately to enhance local
    contrast across all channels without color space conversion. Avoids
    out-of-gamut clipping artifacts that arise from LAB-based approaches.

    Used in retinal vessel segmentation to test contrast enhancement without
    green channel extraction (e.g., Frontiers 2022 multi-channel U-Net pipeline).

    clip_limit=2.0, tile_grid_size=8 are the de facto standard values
    (Liskowski & Krawiec, 2016, IEEE TMI; Wang et al., 2020, IEEE JBHI).

    Input : (H, W, 3) uint8
    Output: (H, W, 3) uint8
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        import cv2
        out = image.copy()
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_grid_size, self._tile_grid_size),
        )
        for ch in range(3):
            out[:, :, ch] = clahe.apply(image[:, :, ch])
        return out

    @property
    def name(self) -> str:
        return f"rgb_clahe(clip={self._clip_limit},tile={self._tile_grid_size})"


class CLAHEStep(PreprocessingStep):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement.

    Applied to a single-channel image to improve local contrast and reveal
    fine vascular structures without amplifying noise (Pizer et al. 1987).

    clip_limit=2.0, tile_grid_size=8 follow common practice in retinal vessel
    segmentation (Liskowski & Krawiec, 2016, IEEE TMI; Wang et al., 2020,
    IEEE JBHI) and match the OpenCV recommended default tile grid.

    Input : (H, W, 1) uint8
    Output: (H, W, 1) uint8
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        import cv2
        single = image[:, :, 0]
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_grid_size, self._tile_grid_size),
        )
        enhanced = clahe.apply(single)
        return np.expand_dims(enhanced, axis=-1)

    @property
    def name(self) -> str:
        return f"clahe(clip={self._clip_limit},tile={self._tile_grid_size})"
