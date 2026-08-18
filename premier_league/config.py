"""Configuration for Premier League win prediction project."""


class PremierLeagueConfig:
    """Configuration settings for the project."""
    
    # Data paths
    csv_path: str = "premier_league_matches.csv"
    model_path: str = "models/premier_league_model.pkl"

    # ML parameters
    test_size: float = 0.2
    random_state: int = 42
    rolling_window: int = 5

    # Model settings
    model_class: str = "LogisticRegression"
    max_iter: int = 1000

    # Features
    features: list = [
        "home_encoded",
        "away_encoded",
        "home_goals_avg",
        "away_goals_avg",
        "home_conceded_avg",
        "away_conceded_avg",
    ]