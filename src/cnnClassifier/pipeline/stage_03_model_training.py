from cnnClassifier import logger
from cnnClassifier.components.model_training import Training
from cnnClassifier.config.configuration import ConfigurationManager


STAGE_NAME = "Model training"


class ModelTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager().get_training_config()
        training = Training(config=config)
        training.get_base_model()
        training.get_train_valid_generator()
        training.train()


if __name__ == "__main__":
    try:
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        ModelTrainingPipeline().main()
        logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)
    except Exception:
        logger.exception("Model training failed")
        raise
