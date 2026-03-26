from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('premier_league_matches.csv')

# first look
print(df.shape)           # how many rows and columns?
print(df.head())          # what does it actually look like?
print(df.dtypes)          # what type is each column?
print(df.isnull().sum())  # any missing values?

# decode FTR into readable labels first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# for numeric features
print(df.describe())

# encode result so we can run the correlation
le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])

# quick correlation check
print(df.corr(numeric_only=True)['result_encoded'].sort_values())

df['HomeGoals'].hist(bins=10)
plt.title('Distribution of Home Goals')
plt.show()

# decode the result column first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# build pre-match rolling averages (what we'd actually know before kickoff)
df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df = df.dropna()

le_home   = LabelEncoder()
le_away   = LabelEncoder()
le_result = LabelEncoder()

df['home_encoded']   = le_home.fit_transform(df['Home'])
df['away_encoded']   = le_away.fit_transform(df['Away'])
df['result_encoded'] = le_result.fit_transform(df['result'])


features = ['home_encoded', 'away_encoded',
            'home_goals_avg', 'away_goals_avg',
            'home_conceded_avg', 'away_conceded_avg']

X = df[features]
y = df['result_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

pred_labels = le_result.inverse_transform(predictions[:10])
print(pred_labels)
