"""Command-line interface for Premier League win prediction."""

import argparse
import logging
import sys
from typing import Optional

from .data import load_data, validate_dataframe
from .features import compute_rolling_averages, encode_categorical
from .model import train_model, evaluate_model, save_model, load_model
from .config import PremierLeagueConfig as project_config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[list] = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Premier League Match Result Predictor"
    )
    parser.add_argument(
        "--csv-path",
        default=project_config.csv_path,
        help="Path to CSV file with match data",
    )
    parser.add_argument(
        "--model-path",
        default=project_config.model_path,
        help="Path to save/load trained model",
    )
    parser.add_argument(
        "--model-type",
        choices=["LogisticRegression", "RandomForest"],
        default="LogisticRegression",
        help="Model type to train",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=project_config.rolling_window,
        help="Rolling average window size",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=project_config.test_size,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=project_config.random_state,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=project_config.max_iter,
        help="Maximum iterations for LogisticRegression",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load existing model for evaluation",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Load and validate data
    df = load_data(args.csv_path)
    df = validate_dataframe(df)

    # Encode target variable
    from sklearn.preprocessing import LabelEncoder
    df["result"] = df["FTR"].map({"H": "Home Win", "A": "Away Win", "D": "Draw"})
    le_result = LabelEncoder()
    df["result_encoded"] = le_result.fit_transform(df["result"])

    # Encode team names
    df, team_encoders = encode_categorical(
        df, ["Home", "Away"], encoder_classes=None
    )

    # Compute rolling averages
    df = compute_rolling_averages(df, window=args.window)

    # Select features
    features = project_config.features
    df = df.dropna(subset=features + ["result_encoded"])
    X = df[features]
    y = df["result_encoded"]

    if args.train:
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(
            X, y,
            model_class=args.model_type,
            max_iter=args.max_iter if args.model_type == "LogisticRegression" else 100,
            random_state=args.random_state,
        )

        # Evaluate
        result = evaluate_model(
            model, X_test, y_test=y_test,
            encoders={"result_encoded": le_result, **team_encoders},
            feature_names=features,
        )

        # Save model
        save_model(model, args.model_path)

        # Print results
        print(f"\n=== Model Results ===")
        print(f"Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"\nClassification Report:")
        from sklearn.metrics import classification_report
        le = le_result
        print(classification_report(
            result.y_test, result.y_pred,
            target_names=list(le.classes_),
            zero_division=0
        ))
    elif args.load_model:
        # Load model and evaluate on current data
        model = load_model(args.model_path)
        print(f"Loaded model: {args.model_path}")
    else:
        print("No action specified. Use --train or --load-model.")


if __name__ == "__main__":
    main()