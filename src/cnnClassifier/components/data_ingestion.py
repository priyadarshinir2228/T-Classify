import zipfile
from pathlib import Path

import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    @staticmethod
    def _is_valid_zip(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
        except (FileNotFoundError, zipfile.BadZipFile):
            return False

    def download_file(self) -> None:
        """Download the dataset only when a valid local archive is unavailable."""
        dataset_url = self.config.source_URL
        zip_download_path = Path(self.config.local_data_file)

        if self._is_valid_zip(zip_download_path):
            logger.info(f"Using existing dataset archive at: {zip_download_path}")
            return

        zip_download_path.parent.mkdir(parents=True, exist_ok=True)
        file_id = dataset_url.split("/")[-2]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        logger.info(f"Downloading data from {dataset_url} into file {zip_download_path}")

        try:
            downloaded_file = gdown.download(download_url, str(zip_download_path), quiet=False)
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Check your internet connection and proxy settings "
                "(HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY)."
            ) from exc

        if downloaded_file is None or not self._is_valid_zip(zip_download_path):
            raise RuntimeError(f"Downloaded dataset archive is invalid: {zip_download_path}")

        logger.info(f"Downloaded data from {dataset_url} into file {zip_download_path}")
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = Path(self.config.unzip_dir)
        zip_path = Path(self.config.local_data_file)

        with zipfile.ZipFile(zip_path) as zip_ref:
            archive_root = next(
                (Path(name).parts[0] for name in zip_ref.namelist() if Path(name).parts),
                None,
            )
            if archive_root and (unzip_path / archive_root).is_dir():
                logger.info(f"Using existing extracted dataset at: {unzip_path / archive_root}")
                return

            unzip_path.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted dataset archive into: {unzip_path}")
