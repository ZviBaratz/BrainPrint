from typing import Tuple

import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import (
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling3D,
    add,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2

tf.disable_v2_behavior()


def generate_resnet_model(
    input_shape: Tuple[int, int, int],
    padding_type: str = "same",
    init_type: str = "he_uniform",
    l2_regularization_factor: float = 0.00005,
    drop_rate: float = 0.2,
    include_context: bool = True,
):
    scan_input = Input(input_shape + (1,), name="T1_Image")

    with tf.name_scope("ResBlock0"):
        inputs = scan_input
        hidden = Conv3D(
            8,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(
            8,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(hidden)
        hidden = BatchNorm(renorm=True)(hidden)
        shortcut = Conv3D(
            8,
            (1, 1, 1),
            strides=(1, 1, 1),
            padding=padding_type,
            kernel_initializer=init_type,
        )(inputs)
        hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding_type
    )(outputs)

    with tf.name_scope("ResBlock1"):
        inputs = pooling
        hidden = Conv3D(
            16,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(
            16,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(hidden)
        hidden = BatchNorm(renorm=True)(hidden)
        shortcut = Conv3D(
            16,
            (1, 1, 1),
            strides=(1, 1, 1),
            padding=padding_type,
            kernel_initializer=init_type,
        )(inputs)
        hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding_type
    )(outputs)

    with tf.name_scope("ResBlock2"):
        inputs = pooling
        hidden = Conv3D(
            32,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(
            32,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(hidden)
        hidden = BatchNorm(renorm=True)(hidden)
        shortcut = Conv3D(
            32,
            (1, 1, 1),
            strides=(1, 1, 1),
            padding=padding_type,
            kernel_initializer=init_type,
        )(inputs)
        hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding_type
    )(outputs)

    with tf.name_scope("ResBlock3"):
        inputs = pooling
        hidden = Conv3D(
            64,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(
            64,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(hidden)
        hidden = BatchNorm(renorm=True)(hidden)
        shortcut = Conv3D(
            64,
            (1, 1, 1),
            strides=(1, 1, 1),
            padding=padding_type,
            kernel_initializer=init_type,
        )(inputs)
        hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding_type
    )(outputs)

    with tf.name_scope("ResBlock4"):
        inputs = pooling
        hidden = Conv3D(
            128,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(inputs)
        hidden = BatchNorm(renorm=True)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(
            128,
            (3, 3, 3),
            padding=padding_type,
            kernel_regularizer=L2(l2_regularization_factor),
            kernel_initializer=init_type,
        )(hidden)
        hidden = BatchNorm(renorm=True)(hidden)
        shortcut = Conv3D(
            128,
            (1, 1, 1),
            strides=(1, 1, 1),
            padding=padding_type,
            kernel_initializer=init_type,
        )(inputs)
        hidden = add([shortcut, hidden])
        outputs = ELU(alpha=1.0)(hidden)

    pooling = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding=padding_type
    )(outputs)

    hidden = Flatten()(pooling)

    hidden = Dense(
        128,
        kernel_regularizer=L2(l2_regularization_factor),
        kernel_initializer=init_type,
        name="FullyConnectedLayer",
    )(hidden)
    hidden = ELU(alpha=1.0)(hidden)
    hidden = Dropout(drop_rate)(hidden)

    if include_context:
        scanner = Input((1,), name="Scanner")
        sex = Input((1,), name="Gender")
        hidden = concatenate([scanner, sex, hidden])

    prediction = Dense(
        1, kernel_regularizer=L2(l2_regularization_factor), name="AgePrediction"
    )(hidden)
    inputs = [scan_input, scanner, sex] if include_context else [scan_input]
    return Model(inputs=inputs, outputs=prediction)
