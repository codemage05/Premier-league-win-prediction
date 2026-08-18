"""Tests for features module."""

import pytest
import pandas as pd
from premier_league.features import compute_rolling_averages, encode_categorical


def test_compute_rolling_averages():
    """Test rolling average computation."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-08-12", "2023-08-19", "2023-08-26"]),
        "Home": ["Arsenal", "Chelsea", "Arsenal"],
        "Away": ["Chelsea", "Arsenal", "Chelsea"],
        "HomeGoals": [2, 1, 3],
        "AwayGoals": [1, 2, 0]
    })
    df = df.sort_values("Date").reset_index(drop=True)
    
    result = compute_rolling_averages(df, window=2)
    assert "home_goals_avg" in result.columns
    assert "away_goals_avg" in result.columns
    assert "home_conceded_avg" in result.columns
    assert "away_conceded_avg" in result.columns
    # No NaN after dropna
    assert result["home_goals_avg"].notna().all()


def test_encode_categorical():
    """Test label encoding."""
    df = pd.DataFrame({
        "team": ["Arsenal", "Chelsea", "Arsenal"],
        "result": ["H", "A", "D"]
    })
    
    encoded, encoders = encode_categorical(df, ["team", "result"])
    assert "team" in encoders
    assert "result" in encoders
    assert len(encoders["team"].classes_) == 2  # Arsenal, Chelsea