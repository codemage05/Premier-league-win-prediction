import os
import warnings

warnings.filterwarnings("ignore", category=ImportWarning)

# Ensure any matplotlib import uses a non-GUI backend
# so GTK-related theme warnings don't appear during headless/script runs.
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('premier_league_matches.csv')

# first look
print(df.shape)           
print(df.head())          
print(df.dtypes)          
print(df.isnull().sum())  


# decode FTR into readable labels first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# for numeric features
print(df.describe())

# encode result so we can run the correlation
from sklearn.preprocessing import LabelEncoder
le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])

# quick correlation check
print(df.corr(numeric_only=True)['result_encoded'].sort_values())


df['HomeGoals'].hist(bins=10)
plt.title('Distribution of Home Goals')
plt.show()