"""Model training, evaluation, and persistence for Premier League prediction."""

import logging
import joblib
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelResult:
    """Container for model training results."""
    
    def __init__(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        encoders: Dict[str, Any],
        metrics: Dict[str, float],
        feature_names: List[str],
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.encoders = encoders
        self.metrics = metrics
        self.feature_names = feature_names


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: str = "LogisticRegression",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Tuple[Any, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Train a classification model.
    
    Args:
        X: Feature DataFrame
        y: Target series
        model_class: Name of model to train ("LogisticRegression" or "RandomForest")
        max_iter: Maximum iterations for LogisticRegression
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    if model_class == "LogisticRegression":
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    elif model_class == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")
    
    model.fit(X_train, y_train)
    logger.info("Trained %s model with %d features", model_class, X.shape[1])
    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoders: Dict[str, Any],
    feature_names: List[str],
) -> ModelResult:
    """Evaluate model and generate metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        encoders: Dict of LabelEncoders for inverse transforming
        feature_names: List of feature names
        
    Returns:
        ModelResult with predictions and metrics
    """
    y_pred = model.predict(X_test)
    
    # Decode predictions and targets for reporting
    le_result = encoders.get("result_encoded")
    if le_result is not None:
        y_test_decoded = le_result.inverse_transform(y_test)
        y_pred_decoded = le_result.inverse_transform(y_pred)
    else:
        y_test_decoded = y_test
        y_pred_decoded = y_pred
    
    accuracy = accuracy_score(y_test, y_pred)
    le_classes = le_result.classes_ if le_result is not None else None
    report = classification_report(
        y_test, y_pred, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=le_classes if le_classes else ["Away Win", "Draw", "Home Win"],
            yticklabels=le_classes if le_classes else ["Away Win", "Draw", "Home Win"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("results/confusion_matrix.png")
        plt.close()
        logger.info("Saved confusion matrix to results/confusion_matrix.png")
    except Exception as e:
        logger.warning("Could not save confusion matrix: %s", e)
    
    metrics = {
        "accuracy": round(accuracy, 4),
    }
    
    if le_result is not None:
        for i, cls in enumerate(le_result.classes_):
            if cls in report:
                metrics[f"precision_{cls}"] = round(report[cls]["precision"], 4)
                metrics[f"recall_{cls}"] = round(report[cls]["recall"], 4)
                metrics[f"f1_{cls}"] = round(report[cls]["f1-score"], 4)
    
    result = ModelResult(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        encoders=encoders,
        metrics=metrics,
        feature_names=feature_names,
    )
    
    logger.info("Model accuracy: %.4f", accuracy)
    return result


def save_model(model: Any, path: str) -> None:
    """Save trained model to disk.
    
    Args:
        model: Trained model to save
        path: File path to save to
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str) -> Any:
    """Load trained model from disk.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded model
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model