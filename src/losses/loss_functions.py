from typing import Callable

import tensorflow as tf
import keras

from config import LOSS_PARAMS


# ── Soft Skeleton (for clDice) ───────────────────────────────────────────────

def _soft_erode(img: tf.Tensor) -> tf.Tensor:
    p1 = -tf.nn.max_pool2d(-img, ksize=(3, 1), strides=(1, 1), padding='SAME')
    p2 = -tf.nn.max_pool2d(-img, ksize=(1, 3), strides=(1, 1), padding='SAME')
    return tf.minimum(p1, p2)


def _soft_dilate(img: tf.Tensor) -> tf.Tensor:
    return tf.nn.max_pool2d(img, ksize=(3, 3), strides=(1, 1), padding='SAME')


def _soft_open(img: tf.Tensor) -> tf.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img: tf.Tensor, iters: int) -> tf.Tensor:
    img1 = _soft_open(img)
    skel = tf.nn.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = tf.nn.relu(img - img1)
        skel = skel + tf.nn.relu(delta - skel * delta)
    return skel


# ── Primitive losses ─────────────────────────────────────────────────────────

def _dice(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float) -> tf.Tensor:
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


def _bce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))


def _ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def _skeleton_recall(y_skel: tf.Tensor, y_pred: tf.Tensor, smooth: float) -> tf.Tensor:
    """Soft recall of prediction over the precomputed tubed skeleton GT.

    Formula: 1 - (Σ(pred × skel) + ε) / (Σ(skel) + ε)

    Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving and
    Resource Efficient Segmentation of Thin Tubular Structures," ECCV 2024.
    """
    inter    = tf.reduce_sum(y_pred * y_skel)
    sum_skel = tf.reduce_sum(y_skel)
    return 1.0 - (inter + smooth) / (sum_skel + smooth)


def _focal(y_true: tf.Tensor, y_pred: tf.Tensor,
           alpha: float, gamma: float, smooth: float) -> tf.Tensor:
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    pos = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
    neg = -(1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)
    return tf.reduce_mean(pos + neg)


def _cldice(y_true: tf.Tensor, y_pred: tf.Tensor,
            smooth: float, iters: int) -> tf.Tensor:
    skel_pred = _soft_skel(y_pred, iters)
    skel_true = _soft_skel(y_true, iters)
    tprec = (tf.reduce_sum(skel_pred * y_true) + smooth) / (tf.reduce_sum(skel_pred) + smooth)
    tsens = (tf.reduce_sum(skel_true * y_pred) + smooth) / (tf.reduce_sum(skel_true) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


# ── MCC Loss ─────────────────────────────────────────────────────────────────

class MCCLoss(keras.losses.Loss):
    def __init__(self, name: str = "mcc_loss"):
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        tp = tf.reduce_sum(y_pred * y_true)
        tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        numerator   = tp * tn - fp * fn
        denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 1.0 - numerator / (denominator + 1e-7)


# ── Combined Loss Functions ──────────────────────────────────────────────────

def bce_mcc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE + MCC — baseline from SA-UNetV2 paper."""
    p = LOSS_PARAMS["bce_mcc"]
    return p["lambda_bce"] * _bce(y_true, y_pred) + p["lambda_mcc"] * MCCLoss()(y_true, y_pred)


def pure_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    p = LOSS_PARAMS["dice"]
    return _dice(y_true, y_pred, smooth=p["smooth"])


def pure_focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    p = LOSS_PARAMS["focal"]
    return _focal(y_true, y_pred, alpha=p["alpha"], gamma=p["gamma"], smooth=p["smooth"])


def cldice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """soft Dice + soft clDice — Shit et al., CVPR 2021.
    L = (1−α)·Dice + α·clDice, α=0.5, smooth=1.0
    """
    p = LOSS_PARAMS["cldice"]
    cl = _cldice(y_true, y_pred, smooth=p["smooth"], iters=p["iters"])
    d  = _dice(y_true, y_pred, smooth=p["smooth"])
    return (1.0 - p["alpha"]) * d + p["alpha"] * cl


def dice_ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice + SSIM — experimental hybrid loss."""
    p = LOSS_PARAMS["dice_ssim"]
    return (p["lambda_dice"] * _dice(y_true, y_pred, smooth=p["smooth"])
            + p["lambda_ssim"] * _ssim(y_true, y_pred))


def bce_ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE + SSIM — ref: Springer 2025, DOI 10.1007/s44443-025-00191-3."""
    p = LOSS_PARAMS["bce_ssim"]
    return p["lambda_bce"] * _bce(y_true, y_pred) + p["lambda_ssim"] * _ssim(y_true, y_pred)


def skel_recall_loss(y_true_combined: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE + Dice + w × Skeleton Recall Loss — DC_SkelREC_and_CE_loss.

    y_true_combined has 2 channels along the last axis:
      [..., :1]  — regular binary GT mask
      [..., 1:2] — precomputed tubed skeleton GT (Kirchhoff et al., ECCV 2024)

    Matches the original DC_SkelREC_and_CE_loss formulation:
      L = L_CE + L_Dice + weight_srec × L_SkelRecall
    """
    p      = LOSS_PARAMS["skel_recall"]
    y_true = y_true_combined[..., :1]
    y_skel = y_true_combined[..., 1:2]
    ce   = _bce(y_true, y_pred)
    d    = _dice(y_true, y_pred, smooth=1e-6)
    srec = _skeleton_recall(y_skel, y_pred, smooth=p["smooth"])
    return ce + d + p["weight_srec"] * srec


# ── Loss Registry ────────────────────────────────────────────────────────────

LOSS_REGISTRY: dict[str, Callable] = {
    "bce_mcc":    bce_mcc_loss,
    "dice":       pure_dice_loss,
    "focal":      pure_focal_loss,
    "cldice":     cldice_loss,
    "dice_ssim":  dice_ssim_loss,
    "bce_ssim":   bce_ssim_loss,
    "skel_recall": skel_recall_loss,
}


def get_loss_function(name: str) -> Callable:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
