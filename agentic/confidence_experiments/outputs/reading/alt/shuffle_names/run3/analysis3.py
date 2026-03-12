import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if col == 'num_words':
        continue
    corr = df['num_words'].corr(df[col])
    print('corr num_words vs', col, corr)

# correlations between possible time columns
for col in ['adjusted_running_time','age','gender','running_time','retake_trial']:
    print(col, 'median', df[col].median(), 'mean', df[col].mean())

