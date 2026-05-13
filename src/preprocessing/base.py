from abc import ABC, abstractmethod

import numpy as np


class PreprocessingStep(ABC):
    """Abstract base for a single preprocessing transformation.

    Contract:
      - Input : (H, W, C) numpy array, dtype uint8, range [0, 255]
      - Output: (H, W, C') numpy array, dtype uint8, range [0, 255]
                C' may differ from C (e.g. GreenChannelStep: 3 → 1)
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
