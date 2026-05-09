import keras
from keras import ops
from keras.layers import Permute, Concatenate, Conv2D, multiply


class _ChannelPool(keras.layers.Layer):
    """Reduce channel axis (axis=-1) with keepdims=True — mean or max."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        assert mode in ("mean", "max")
        self.mode = mode

    def call(self, x):
        if self.mode == "mean":
            return ops.mean(x, axis=-1, keepdims=True)
        return ops.max(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def spatial_attention(input_feature):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = _ChannelPool("mean")(cbam_feature)
    max_pool = _ChannelPool("max")(cbam_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False,
    )(concat)

    return multiply([input_feature, cbam_feature])


def csa_block(encoder_feat, decoder_feat, kernel_size=7):
    """Cross Spatial Attention between encoder and decoder feature maps."""
    avg1 = _ChannelPool("mean")(encoder_feat)
    avg2 = _ChannelPool("mean")(decoder_feat)

    sa = Concatenate(axis=-1)([avg1, avg2])
    sa = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False,
    )(sa)

    return multiply([encoder_feat, sa])
