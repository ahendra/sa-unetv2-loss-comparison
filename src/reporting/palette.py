"""Shared colour palette for all report charts.

Import in any reporter:
    from .palette import LOSS_COLORS, METRIC_COLORS, METRIC_LABELS
"""

# ── Loss-function colours ─────────────────────────────────────────────────────
# One distinct colour per loss key, used consistently across every chart that
# compares loss functions (radar, bar, heatmap, …).
LOSS_COLORS = {
    "bce_mcc":   "#2176AE",   # steel blue
    "dice":      "#3BB273",   # emerald green
    "focal":     "#E84855",   # vivid coral
    "cldice":    "#F4A100",   # amber
    "dice_ssim": "#7B2D8B",   # purple
    "bce_ssim":  "#00B4D8",   # sky blue
}

# ── Metric colours ────────────────────────────────────────────────────────────
# One colour per evaluation metric — used to tint subplot titles so readers
# can identify the same metric instantly across different chart types.
METRIC_COLORS = {
    "accuracy":     "#4E79A7",   # muted blue
    "sensitivity":  "#F28E2B",   # warm orange
    "specificity":  "#E15759",   # soft coral-red
    "auc":          "#59A14F",   # forest green
    "mcc":          "#76B7B2",   # teal
    "f1":           "#B07AA1",   # lavender purple
    "jaccard":      "#EDC948",   # golden yellow
    "cldice":       "#FF9DA7",   # rose pink
    "betti0_error": "#9C755F",   # warm brown
    "betti1_error": "#BAB0AC",   # warm gray
}

# ── Metric display labels ─────────────────────────────────────────────────────
METRIC_LABELS = {
    "accuracy":     "Accuracy",
    "sensitivity":  "Sensitivity",
    "specificity":  "Specificity",
    "auc":          "AUC",
    "mcc":          "MCC",
    "f1":           "F1",
    "jaccard":      "Jaccard",
    "cldice":       "clDice",
    "betti0_error": "Betti-0 Error (β₀)\n(↓ lebih kecil = lebih baik)",
    "betti1_error": "Betti-1 Error (β₁)\n(↓ lebih kecil = lebih baik)",
}
