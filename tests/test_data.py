"""Tests for data module."""

import pytest
import pandas as pd
import tempfile
import os
from premier_league.data import load_data, validate_dataframe, DataValidationError


def test_load_data_valid():
    """Test loading valid CSV data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Date,Home,Away,HomeGoals,AwayGoals,FTR\n")
        f.write("2023-08-12,Arsenal,Chelsea,2,1,H\n")
        f.write("2023-08-19,Chelsea,Arsenal,1,2,A\n")
        tmp = f.name
    
    try:
        df = load_data(tmp)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    finally:
        os.unlink(tmp)


def test_load_data_missing_columns():
    """Test that missing columns raise DataValidationError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Date,Home,Away\n")
        f.write("2023-08-12,Arsenal,Chelsea\n")
        tmp = f.name
    
    try:
        with pytest.raises(DataValidationError):
            load_data(tmp)
    finally:
        os.unlink(tmp)


def test_validate_dataframe():
    """Test dataframe validation and sorting."""
    df = pd.DataFrame({
        "Date": ["2023-08-19", "2023-08-12"],  # unsorted
        "Home": ["Arsenal", "Chelsea"],
        "Away": ["Chelsea", "Arsenal"],
        "HomeGoals": [1, 2],
        "AwayGoals": [2, 1],
        "FTR": ["A", "H"]
    })
    df["Date"] = pd.to_datetime(df["Date"])
    cleaned = validate_dataframe(df)
    assert len(cleaned) == 2
    # Should be sorted by date
    assert cleaned["Date"].iloc[0] <= cleaned["Date"].iloc[1]