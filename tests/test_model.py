"""Tests for model module."""

import pytest
import pandas as pd
import numpy as np
from premier_league.model import train_model, evaluate_model


def test_train_logistic_regression():
    """Test training a logistic regression model."""
    X = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 2,
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 10 elements to match X
    
    model, X_train, X_test, y_train, y_test = train_model(
        X, y, model_class="LogisticRegression", max_iter=1000
    )
    assert model is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_evaluate_model():
    """Test model evaluation."""
    X = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20),
    })
    y = pd.Series(np.random.randint(0, 3, 20))  # 3 classes, 20 elements to match X
    
    model, X_train, X_test, y_train, y_test = train_model(X, y, model_class="LogisticRegression")
    result = evaluate_model(model, X_test, y_test=y_test,
                           encoders={"result_encoded": None},
                           feature_names=["feature1", "feature2"])
    
    assert "accuracy" in result.metrics
    assert 0 <= result.metrics["accuracy"] <= 1