import keras
import tensorflow as tf


class DropBlock2D(keras.layers.Layer):
    """
    DropBlock regularization for 2D feature maps (Ghiasi et al., 2018).

    Drop-in replacement for keras_cv.layers.DropBlock2D, which is broken
    under Keras 3.x / TF 2.19 (SymbolicTensor passed to int()).

    Matches the original keras_cv behavior:
      - Independent drop mask per channel (shape: batch, H, W, C)
      - Gamma computed from DropBlock paper eq. 6
      - Per-sample normalization to preserve expected activation magnitude
    """

    def __init__(self, block_size: int = 7, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate == 0.0:
            return x

        # Use static spatial dims (always concrete for fixed input sizes)
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]
        if h is None or w is None or c is None:
            return x

        # Gamma: seed drop probability (DropBlock paper, eq. 6)
        gamma = (self.rate * h * w) / (
            self.block_size ** 2
            * max(h - self.block_size + 1, 1)
            * max(w - self.block_size + 1, 1)
        )
        gamma = min(gamma, 1.0)

        # Independent mask per channel: shape (batch, H, W, C)
        batch = tf.shape(x)[0]
        seed = tf.random.uniform(shape=(batch, h, w, c), dtype=tf.float32)
        seed_mask = tf.cast(seed < gamma, dtype=tf.float32)

        # Expand seeds to blocks via max-pool (applied over H×W, independent per C)
        # max_pool2d requires (batch, H, W, C)
        block_mask = tf.nn.max_pool2d(
            seed_mask,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
        )

        # Invert: 1 = keep, 0 = drop
        block_mask = 1.0 - block_mask

        # Per-sample normalisation to preserve expected activation magnitude
        # Sum kept positions per sample: shape (batch, 1, 1, 1)
        n_total = tf.cast(h * w * c, tf.float32)
        n_kept = tf.reduce_sum(block_mask, axis=[1, 2, 3], keepdims=True) + 1e-8

        return x * block_mask * (n_total / n_kept)

    def get_config(self):
        return {**super().get_config(), "block_size": self.block_size, "rate": self.rate}
