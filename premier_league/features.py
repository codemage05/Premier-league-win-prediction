"""Feature engineering for Premier League match data."""

import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


def compute_rolling_averages(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Compute pre-match rolling averages for each team.
    
    Uses shift(1) to prevent data leakage - only uses past matches.
    
    Args:
        df: DataFrame with match data sorted by Date
        window: Rolling window size in matches
        
    Returns:
        DataFrame with added rolling average features
    """
    df = df.copy()
    
    # Home team rolling averages
    df["home_goals_avg"] = df.groupby("Home")["HomeGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["away_goals_avg"] = df.groupby("Away")["AwayGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["home_conceded_avg"] = df.groupby("Home")["AwayGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["away_conceded_avg"] = df.groupby("Away")["HomeGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    n_before = df.isna().any(axis=1).sum()
    if n_before > 0:
        logger.warning(
            "Dropping %d rows due to NaN in rolling features", n_before
        )
    df = df.dropna(subset=["home_goals_avg", "away_goals_avg", 
                          "home_conceded_avg", "away_conceded_avg"])
    
    logger.info("Computed rolling averages (window=%d) for %d matches", window, len(df))
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    encoder_classes: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Label encode categorical columns, creating new _encoded columns.
    
    Args:
        df: DataFrame to encode
        columns: List of column names to encode
        encoder_classes: Optional dict of pre-fit encoder classes
        
    Returns:
        Tuple of (encoded DataFrame, dict of LabelEncoders)
    """
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in columns:
        le = LabelEncoder()
        if encoder_classes and col in encoder_classes:
            le.classes_ = encoder_classes[col]
        df_encoded[f"{col.lower()}_encoded"] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        logger.info("Encoded column %s with %d classes", col, len(le.classes_))
    
    return df_encoded, encoders