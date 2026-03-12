import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate word count columns: retake_trial
word_col = 'retake_trial'
# candidate time columns
cand_time_cols = ['adjusted_running_time', 'age', 'gender']

for time_col in cand_time_cols:
    speed = df[word_col] / (df[time_col] / 60000.0)
    corr = speed.corr(df['running_time'])
    print(time_col, 'corr with running_time', corr, 'speed mean', speed.mean(), 'speed max', speed.max())

# examine distributions for columns
cols = df.columns
for col in cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(col, df[col].min(), df[col].max(), df[col].mean(), df[col].std(), df[col].nunique())
    else:
        print(col, df[col].nunique(), df[col].dropna().unique()[:5])
