import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


EXPECTED_CLASS_INDICES = {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self) -> None:
        """Load a saved model without its serialized compile state."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False,
        )
        self._compile_model(self.config.params_learning_rate)

    def _compile_model(self, learning_rate: float) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_train_valid_generator(self) -> None:
        generator_kwargs = {
            "rescale": 1.0 / 255,
            "validation_split": 0.2,
        }
        if self.config.params_augmentation:
            generator_kwargs.update(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
            )

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **generator_kwargs
        )
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
        )
        flow_kwargs = {
            "directory": str(self.config.training_data),
            "target_size": tuple(self.config.params_image_size[:2]),
            "batch_size": self.config.params_batch_size,
            "class_mode": "categorical",
            "classes": list(EXPECTED_CLASS_INDICES),
        }
        self.train_generator = train_datagen.flow_from_directory(
            subset="training", shuffle=True, **flow_kwargs
        )
        self.valid_generator = valid_datagen.flow_from_directory(
            subset="validation", shuffle=False, **flow_kwargs
        )

        print(self.train_generator.class_indices)
        if self.train_generator.class_indices != EXPECTED_CLASS_INDICES:
            raise ValueError(
                "Expected four classes Cyst, Normal, Stone, Tumor; found "
                f"{self.train_generator.class_indices}"
            )
        logger.info("Verified class indices: %s", self.train_generator.class_indices)

    def train(self) -> None:
        self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
        )

        if self.config.params_fine_tune:
            self.fine_tune()

        self.model.save(self.config.trained_model_path)
        logger.info("Trained model saved to: %s", self.config.trained_model_path)

    def fine_tune(self) -> None:
        """Unfreeze the final four VGG16 layers and train at a lower rate."""
        # The saved functional model contains the VGG16 layers followed by
        # Flatten and Dense classification layers.
        for layer in self.model.layers[:-2][-4:]:
            layer.trainable = True

        self._compile_model(learning_rate=1e-5)
        self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
        )
