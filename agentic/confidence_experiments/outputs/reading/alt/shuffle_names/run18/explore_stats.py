import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

df = pd.read_csv('reading.csv')

num_cols = df.select_dtypes(include=['number']).columns
print('Numeric columns:', num_cols.tolist())

summary = df[num_cols].describe().T[['count','mean','std','min','25%','50%','75%','max']]
print('\nSummary:')
print(summary)

# correlations with num_words and adjusted_running_time and running_time
for col in num_cols:
    if col in ['num_words','adjusted_running_time','running_time','age','gender']:
        continue

# pairwise correlations of key time columns
key_cols = ['adjusted_running_time','running_time','age','gender','num_words']
key_cols = [c for c in key_cols if c in num_cols]
print('\nKey correlations:')
print(df[key_cols].corr())

# show value counts for some categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print('\n', col)
    print(df[col].value_counts(dropna=False).head(10))
