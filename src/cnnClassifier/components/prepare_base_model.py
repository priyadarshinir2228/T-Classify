import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        learning_rate: float,
        freeze_all: bool = True,
        freeze_till: int | None = None,
    ) -> tf.keras.Model:
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(classes, activation="softmax")(flatten_in)
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return full_model

    def get_base_model(self) -> None:
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )
        self.model.save(self.config.base_model_path)
        logger.info("VGG16 base model created")

    def update_base_model(self) -> None:
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate,
        )
        self.full_model.save(self.config.updated_base_model_path)
        logger.info("Updated base model created")
