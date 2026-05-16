from .pipeline import PreprocessingPipeline
from .steps import CLAHEStep, GreenChannelStep, IdentityStep, RGBCLAHEStep

VALID_MODES = ("rgb", "clahe", "green", "green_clahe")


def build_pipeline(
    mode: str,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> PreprocessingPipeline:
    """Factory that builds a PreprocessingPipeline from a mode string.

    mode='rgb'         → IdentityStep                    (output channels: 3)
    mode='clahe'       → RGBCLAHEStep (per-channel RGB)   (output channels: 3)
    mode='green'       → GreenChannelStep                 (output channels: 1)
    mode='green_clahe' → GreenChannelStep + CLAHEStep     (output channels: 1)

    Callers depend on this abstraction, not on concrete step classes.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown preprocessing_mode '{mode}'. Valid options: {VALID_MODES}"
        )

    if mode == "rgb":
        return PreprocessingPipeline([IdentityStep()])
    if mode == "clahe":
        return PreprocessingPipeline([
            RGBCLAHEStep(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid),
        ])
    if mode == "green":
        return PreprocessingPipeline([GreenChannelStep()])
    return PreprocessingPipeline([
        GreenChannelStep(),
        CLAHEStep(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid),
    ])
