from typing import Optional

import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras_tuner.applications import HyperXception
from keras_tuner.tuners import Hyperband
from tensorflow_addons.metrics import RSquare

from brainprint.dl.load import read_dataset
from brainprint.dl.utils import INFO_SHAPE, VOLUME_SHAPE


class HyperBinaryXception(HyperXception):
    def __init__(self, input_shape, target: str, **kwargs):
        input_tensor = tfl.Input(input_shape, name="volume")
        self.target = target
        super().__init__(
            input_tensor=input_tensor, include_top=False, **kwargs
        )

    def build(self, hp):
        # Build the ResNet model.
        xception = super().build(hp)

        # Extract the last layer from the ResNet model.
        xception_output_layer = xception.layers[-1].output

        # Concatenate the ResNet output with the info input.
        info_input = tfl.Input(INFO_SHAPE, name="acquisition_parameters")
        X = tfl.concatenate([xception_output_layer, info_input])

        if self.target == "sex":
            # Add a fully connected binary classification layer.
            prediction_layer = tfl.Dense(
                1, activation="sigmoid", name="prediction_layer"
            )(X)
        elif self.target == "age":
            # Add a fully connected regression layer.
            prediction_layer = tfl.Dense(
                1, activation="linear", name="prediction_layer"
            )(X)
        elif self.target == "personality":
            # Add a fully connected regression layer.
            prediction_layer = tfl.Dense(
                5, activation="linear", name="prediction_layer"
            )(X)

        # Create the model.
        model = tf.keras.Model(
            inputs=[xception.inputs, info_input],
            outputs=prediction_layer,
            name="BinaryXception",
        )

        # Compile the model.
        optimizer = tf.keras.optimizers.get("adam")
        optimizer.learning_rate = hp.Choice(
            "learning_rate", [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5], default=5e-4
        )
        if self.target == "personality":
            loss = tf.keras.losses.MeanSquaredError()
        elif self.target == "age":
            loss = tf.keras.losses.MeanAbsoluteError()
        elif self.target == "sex":
            loss = tf.keras.losses.BinaryCrossentropy()

        metrics = (
            ["accuracy"] if self.target == "sex" else ["mae", "mse", RSquare()]
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return model


def run_hyper_xception(
    max_epochs: int = 100,
    executions_per_trial: int = 2,
    factor: int = 3,
    hyperband_iterations: int = 2,
    epochs: int = 50,
    directory: str = "HyperXception",
    early_stopping_patience: Optional[int] = 4,
    target: str = "sex",
    project_name: Optional[str] = None,
    run: bool = True,
    seed: int = 0,
):
    # Mediate input arguments.
    target = target.lower()
    project_name = project_name or target.capitalize()

    model = HyperBinaryXception(input_shape=VOLUME_SHAPE, target=target)
    if target == "personality":
        objective = "val_mse"
    elif target == "age":
        objective = "val_mae"
    elif target == "sex":
        objective = "val_accuracy"
    tuner = Hyperband(
        model,
        objective=objective,
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        executions_per_trial=executions_per_trial,
        seed=seed,
        project_name=project_name,
        directory=directory,
    )
    if not run:
        return tuner
    train, test = read_dataset(
        include_info=True, one_hot_encode=False, target=target
    )
    callbacks = []
    if early_stopping_patience is not None:
        stop_early = tf.keras.callbacks.EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
        callbacks.append(stop_early)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"{directory}/{project_name}/logs",
        histogram_freq=1,
        write_graph=True,
    )
    callbacks.append(tensorboard)
    return tuner.search(
        train,
        validation_data=test,
        epochs=epochs,
        callbacks=callbacks,
    )
