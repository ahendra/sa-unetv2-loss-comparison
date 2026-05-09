from typing import Tuple

import keras
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    Activation, concatenate, GroupNormalization,
)
from keras.models import Model
from .dropblock import DropBlock2D
from .attention import spatial_attention, csa_block


def _conv_bn_act(x, filters: int, block_size: int, rate: float):
    x = Conv2D(filters, (3, 3), padding="same", activation=None,
               kernel_initializer='he_normal')(x)
    x = DropBlock2D(block_size=block_size, rate=rate)(x)
    x = GroupNormalization(groups=8, axis=-1)(x)
    x = Activation('silu')(x)
    return x


def _encoder_block(x, filters: int, block_size: int, rate: float):
    x = _conv_bn_act(x, filters, block_size, rate)
    x = _conv_bn_act(x, filters, block_size, rate)
    return x


def build_sa_unetv2(
    input_size: Tuple[int, int, int] = (592, 592, 3),
    start_neurons: int = 16,
    block_size: int = 7,
    rate: float = 0.15,
) -> Model:
    """
    SA-UNetV2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation.

    Architecture:
    - Encoder: 3 levels with DropBlock2D + GroupNorm + SiLU
    - Bottleneck: Spatial Attention (CBAM-style)
    - Decoder: Cross Spatial Attention (CSA) skip connections
    - Output: 1-channel sigmoid segmentation map
    """
    inputs = Input(input_size)

    # Encoder
    conv1 = _encoder_block(inputs, start_neurons * 1, block_size, rate)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = _encoder_block(pool1, start_neurons * 2, block_size, rate)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = _encoder_block(pool2, start_neurons * 3, block_size, rate)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Bottleneck with Spatial Attention
    convm = _conv_bn_act(pool3, start_neurons * 4, block_size, rate)
    convm = spatial_attention(convm)
    convm = _conv_bn_act(convm, start_neurons * 4, block_size, rate)

    # Decoder with CSA skip connections
    deconv3 = Conv2DTranspose(start_neurons * 3, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, csa_block(conv3, deconv3)])
    uconv3 = _encoder_block(uconv3, start_neurons * 3, block_size, rate)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, csa_block(conv2, deconv2)])
    uconv2 = _encoder_block(uconv2, start_neurons * 2, block_size, rate)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, csa_block(conv1, deconv1)])
    uconv1 = _encoder_block(uconv1, start_neurons * 1, block_size, rate)

    output = Conv2D(1, (1, 1), padding="same", activation=None,
                    kernel_initializer='he_normal')(uconv1)
    output = Activation('sigmoid')(output)

    return Model(inputs, output, name="SA_UNetV2")
