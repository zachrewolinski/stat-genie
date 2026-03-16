import pandas as pd
import numpy as np
import json

path='soccer.csv'

df=pd.read_csv(path)

# basic summary for numeric columns
num_cols=df.select_dtypes(include=[np.number]).columns
summary=df[num_cols].describe().T
print('Numeric summary')
print(summary)

# count unique values for key columns
for col in ['rater1','nExp','rater2','yellowCards','yellowReds','redCards']:
    if col in df.columns:
        print('\n', col, df[col].min(), df[col].max(), df[col].mean())

# look at correlations with red/yellow columns and games-like column
# assume games column maybe 'redCards' (range 1-47), also check correlations
print('\nCorrelations with potential games column redCards')
print(df[['redCards','yellowCards','yellowReds']].corr())

# check distribution of rater1 and nExp
print('\nRater1 value counts')
print(df['rater1'].value_counts(dropna=False).sort_index())
print('\nNExp value counts')
print(df['nExp'].value_counts(dropna=False).sort_index())

# compute mean skin tone
skin=df[['rater1','nExp']].mean(axis=1)

# define light vs dark threshold at 0.5
light=skin<=0.5

# candidate outcomes: yellowCards or yellowReds?
for outcome in ['yellowCards','yellowReds']:
    if outcome in df.columns:
        # compare mean outcome between light and dark
        light_mean=df.loc[light, outcome].mean()
        dark_mean=df.loc[~light, outcome].mean()
        print('\nOutcome', outcome)
        print('light_mean', light_mean, 'dark_mean', dark_mean)
