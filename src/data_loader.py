import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_FILENAME,
    DATASET_URL,
    DATASET_ZIP,
    EXPECTED_COLUMNS,
    RAW_DATA_DIR,
)
from src.utils import setup_logging, validate_columns


logger = setup_logging("data_loader")


def download_dataset(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / DATASET_ZIP
    csv_path = dest_dir / DATASET_FILENAME
    if csv_path.exists():
        logger.info("Dataset already exists at %s, skipping download", csv_path)
        return csv_path
    logger.info("Downloading dataset from %s", url)
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Downloaded dataset to %s", zip_path)
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise RuntimeError(f"Failed to download dataset from {url}") from exc
    logger.info("Extracting zip archive")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("Extracted dataset to %s", dest_dir)
    except Exception as exc:
        logger.error("Failed to extract dataset: %s", exc)
        raise RuntimeError("Failed to extract dataset archive") from exc
    if not csv_path.exists():
        extracted_files = list(dest_dir.glob("*.csv"))
        if extracted_files:
            extracted_files[0].rename(csv_path)
        else:
            raise FileNotFoundError(f"No CSV file found after extraction in {dest_dir}")
    return csv_path


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    if data_path is None:
        data_path = RAW_DATA_DIR / DATASET_FILENAME
    if not data_path.exists():
        logger.info("Dataset not found at %s, downloading...", data_path)
        data_path = download_dataset(DATASET_URL, RAW_DATA_DIR)
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded dataset with shape %s", df.shape)
    validate_columns(df, EXPECTED_COLUMNS)
    logger.info("Dataset validation passed: all expected columns present")
    return df


__all__ = ["download_dataset", "load_dataset"]
