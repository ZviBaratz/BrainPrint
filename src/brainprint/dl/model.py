from typing import Optional

import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.regularizers import L2

from brainprint.dl.load import read_dataset

VOLUME_SHAPE = (91, 109, 91)
INFO_SHAPE = (7,)
FILTER_SIZES = (8, 16, 32, 64)


def create_convolution_layer(
    inputs,
    filters,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    kernel_initializer: str = "he_uniform",
    kernel_regularizer=L2(0.0005),
    batch_normalization: bool = True,
    activation: Optional[str] = "relu",
    dropout: Optional[float] = 0.2,
):
    inputs = tfl.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )(inputs)
    if batch_normalization:
        inputs = tfl.BatchNormalization()(inputs)
    if activation is not None:
        inputs = tfl.Activation(activation)(inputs)
        if dropout is not None:
            inputs = tfl.Dropout(dropout)(inputs)
    return inputs


def create_residual_block(
    inputs,
    filters,
    n_layers: int = 2,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    kernel_initializer: str = "he_uniform",
    kernel_regularizer=L2(0.0005),
    batch_normalization: bool = True,
    activation: str = "relu",
):
    hidden = inputs
    for i in range(n_layers):
        last_layer = i == n_layers - 1
        layer_activation = None if last_layer else activation
        hidden = create_convolution_layer(
            hidden,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            batch_normalization=batch_normalization,
            activation=layer_activation,
        )
        if last_layer:
            shortcut = tfl.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,
            )(inputs)
            shortcut = tfl.BatchNormalization()(shortcut)
            hidden = tfl.add([shortcut, hidden])
    return tfl.Activation(activation)(hidden)


def create_model():
    volume_input = tfl.Input(VOLUME_SHAPE, name="volume")
    info_input = tfl.Input(INFO_SHAPE, name="info")

    X = volume_input
    for i, filter_size in enumerate(FILTER_SIZES):
        with tf.name_scope(f"ResBlock{i}"):
            X = create_residual_block(X, filters=filter_size)
        X = tfl.MaxPooling2D(pool_size=2, strides=2, padding="same")(X)

    # Flatten and add dense layers.
    X = tfl.Flatten()(X)
    X = tfl.Dense(128, activation="relu")(X)
    X = tfl.Dropout(0.4)(X)
    # Concatenate contextual info.
    X = tfl.concatenate([X, info_input])
    # Output layer.
    X = tfl.Dense(1, activation="sigmoid")(X)

    model = tf.keras.Model(inputs=[volume_input, info_input], outputs=X)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def run_model(n_epochs: int = 20):
    train, test = read_dataset(include_info=True)
    model = create_model()
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    return model.fit(
        train, epochs=n_epochs, validation_data=test, callbacks=[stop_early]
    )
