"""Data loading and validation for Premier League match data."""

import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and perform basic validation.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Loaded and validated DataFrame
        
    Raises:
        DataValidationError: If required columns are missing
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error("CSV file not found at: %s", csv_path)
        raise

    required_columns = {"Date", "Home", "Away", "HomeGoals", "AwayGoals", "FTR"}
    missing = required_columns - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Validate FTR values
    valid_results = {"H", "A", "D"}
    invalid_ftr = ~df["FTR"].isin(valid_results)
    if invalid_ftr.any():
        logger.warning("Found %d rows with invalid FTR values", invalid_ftr.sum())

    logger.info("Loaded dataset: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Additional validation and cleaning steps.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Cleaned DataFrame
    """
    # Remove rows with NaT dates
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df