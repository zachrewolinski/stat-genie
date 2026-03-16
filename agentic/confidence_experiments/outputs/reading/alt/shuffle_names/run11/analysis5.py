import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

# potential time columns (ms)
# compute wpm from adjusted_running_time and num_words
wpm_adj = df['num_words'] / df['adjusted_running_time'] * 60000
wpm_age = df['num_words'] / df['age'] * 60000

candidates = ['running_time','gender','education','retake_trial','Flesch_Kincaid','uuid']
print('wpm_adj stats', wpm_adj.describe())
print('wpm_age stats', wpm_age.describe())

for col in candidates:
    if col in df.columns:
        for name, wpm in [('wpm_adj', wpm_adj), ('wpm_age', wpm_age)]:
            corr = df[col].corr(wpm)
            print(f'corr {col} vs {name}: {corr}')
