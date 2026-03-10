import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# show some rows to compare columns
cols = ['adjusted_running_time','age','gender','running_time','retake_trial','num_words']
print(df[cols].head())

# compute correlations between numeric columns for clues
num_cols = df.select_dtypes(include=['number']).columns
corr = df[num_cols].corr()
print('\nTop correlations with running_time:')
print(corr['running_time'].sort_values(ascending=False).head(10))

# check ratio adjusted_running_time/running_time
ratio = df['adjusted_running_time'] / df['running_time']
print('\nRatio adjusted_running_time/running_time summary:')
print(ratio.describe())

# check ratio age/running_time
ratio2 = df['age'] / df['running_time']
print('\nRatio age/running_time summary:')
print(ratio2.describe())

# look at unique values of some categorical columns
cat_cols = ['english_native','page_id','reader_view','img_width']
for c in cat_cols:
    print('\n', c, df[c].value_counts(dropna=False).head())
