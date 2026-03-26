# ⚽ Premier League Win Prediction

<div align="center">

![Premier League](https://img.shields.io/badge/Premier%20League-ML%20Project-purple?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A machine learning model that predicts the outcome of English Premier League matches using historical match data and rolling team performance statistics.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Features Used](#-features-used)
- [Model](#-model)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🧠 Overview

This project uses **Logistic Regression** to predict whether a Premier League match will result in a **Home Win**, **Away Win**, or **Draw** — based purely on pre-match statistics that would realistically be available before kickoff.

Rather than leaking post-match data into predictions, this model builds **rolling 5-match averages** for each team (goals scored, goals conceded) to simulate real-world prediction conditions.

> 🚀 This is my **first end-to-end ML project** — from raw CSV data to a trained classifier with a confusion matrix.

---

## ⚙️ How It Works

```
Raw CSV Data
     │
     ▼
Data Cleaning & Parsing
     │
     ▼
Feature Engineering
(Rolling 5-match team averages — goals scored & conceded)
     │
     ▼
Label Encoding
(Teams → integers, FTR → Home Win / Away Win / Draw)
     │
     ▼
Train / Test Split (80/20)
     │
     ▼
Logistic Regression Model
     │
     ▼
Evaluation (Accuracy, Classification Report, Confusion Matrix)
```

---

## 📁 Project Structure

```
Premier-league-win-prediction/
│
├── PL.py                         # Main ML script
├── premier_league_matches.csv    # Historical match dataset
└── README.md                     # You are here
```

---

## 📊 Dataset

The dataset `premier_league_matches.csv` contains historical English Premier League match records.

| Column       | Description                          |
|--------------|--------------------------------------|
| `Date`       | Match date                           |
| `Home`       | Home team name                       |
| `Away`       | Away team name                       |
| `HomeGoals`  | Goals scored by the home team        |
| `AwayGoals`  | Goals scored by the away team        |
| `FTR`        | Full-Time Result (`H`, `A`, `D`)     |

The `FTR` column is decoded into human-readable labels:

| Code | Meaning   |
|------|-----------|
| `H`  | Home Win  |
| `A`  | Away Win  |
| `D`  | Draw      |

---

## 🔧 Features Used

To avoid data leakage, only **pre-match features** are used for prediction:

| Feature              | Description                                          |
|----------------------|------------------------------------------------------|
| `home_encoded`       | Label-encoded home team ID                          |
| `away_encoded`       | Label-encoded away team ID                          |
| `home_goals_avg`     | Home team's rolling avg goals scored (last 5 games) |
| `away_goals_avg`     | Away team's rolling avg goals scored (last 5 games) |
| `home_conceded_avg`  | Home team's rolling avg goals conceded (last 5)     |
| `away_conceded_avg`  | Away team's rolling avg goals conceded (last 5)     |

> Rolling averages are computed using `.shift(1)` to ensure **no future data leaks into training**.

---

## 🤖 Model

| Detail          | Value                    |
|-----------------|--------------------------|
| Algorithm       | Logistic Regression      |
| Library         | `scikit-learn`           |
| Train/Test Split| 80% / 20%                |
| Max Iterations  | 1000                     |
| Random State    | 42                       |

---

## 📈 Results

The model outputs:

- **Accuracy Score** — overall prediction correctness
- **Classification Report** — precision, recall, and F1-score per class
- **Confusion Matrix** — visual breakdown of actual vs predicted outcomes

Example confusion matrix output:

```
              Predicted
              Away Win  Draw  Home Win
Actual Away Win   [ ✓ ]  [ ]   [ ]
       Draw       [ ]    [ ✓ ] [ ]
       Home Win   [ ]    [ ]   [ ✓ ]
```

*(Actual values depend on your dataset run)*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/codemage05/Premier-league-win-prediction.git
cd Premier-league-win-prediction
```

### 2. Install dependencies

```bash
pip install pandas matplotlib scikit-learn seaborn
```

### 3. Run the model

```bash
python PL.py
```

---

## 📦 Requirements

```
pandas
matplotlib
scikit-learn
seaborn
```

You can also install via:

```bash
pip install pandas matplotlib scikit-learn seaborn
```

> Python 3.8 or higher is recommended.

---

## 🖥️ Usage

After running `PL.py`, you will see:

1. **Dataset overview** — shape, column types, value counts
2. **Correlation analysis** — how each feature relates to the result
3. **Home Goals histogram** — distribution plot
4. **Model accuracy** printed to console
5. **Classification report** — per-class precision & recall
6. **Confusion matrix heatmap** — rendered via seaborn

---

## 🖼️ Visualizations

The script automatically generates:

- 📊 **Home Goals Distribution** — histogram showing how often teams score at home
- 🔥 **Confusion Matrix Heatmap** — color-coded grid showing model performance across all 3 outcomes

---

## 🗺️ Roadmap

Future improvements planned for this project:

- [ ] Add more features (shots on target, possession %, form streaks)
- [ ] Try Random Forest and XGBoost classifiers
- [ ] Cross-validation for more robust accuracy estimates
- [ ] Build a simple CLI or web interface to predict upcoming fixtures
- [ ] Add player-level data (injuries, top scorer form)

---

## 👤 Author

**codemage05**

> Built as a first ML project — combining a love for football and data science.

[![GitHub](https://img.shields.io/badge/GitHub-codemage05-black?style=flat-square&logo=github)](https://github.com/codemage05)

---

<div align="center">

⭐ **If you found this useful, consider starring the repo!** ⭐

*Made with ❤️ and Python*

</div>
