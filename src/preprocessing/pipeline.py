from typing import List

import numpy as np

from .base import PreprocessingStep


class PreprocessingPipeline:
    """Ordered chain of PreprocessingStep objects.

    Open/Closed: extend by composing new steps; this class never changes.
    """

    def __init__(self, steps: List[PreprocessingStep]):
        self._steps = steps

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply all steps in order. Input must be (H, W, 3) uint8."""
        result = image
        for step in self._steps:
            result = step.apply(result)
        return result

    @property
    def output_channels(self) -> int:
        dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        return self.apply(dummy).shape[-1]

    @property
    def description(self) -> str:
        return " → ".join(s.name for s in self._steps)
