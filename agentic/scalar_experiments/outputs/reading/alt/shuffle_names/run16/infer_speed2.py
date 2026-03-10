import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

adjusted_time = df['age']  # inferred
num_words = df['num_words']
# compute wpm using adjusted_time in ms
wpm_adj = num_words * 60000 / adjusted_time
wpm_total = num_words * 60000 / df['adjusted_running_time']

print('wpm_adj summary', wpm_adj.describe())
print('wpm_total summary', wpm_total.describe())

for col in ['running_time','uuid','Flesch_Kincaid','retake_trial']:
    if df[col].dtype.kind in 'if':
        print(col, 'corr wpm_adj', np.corrcoef(wpm_adj, df[col])[0,1], 'corr wpm_total', np.corrcoef(wpm_total, df[col])[0,1])

# check if running_time is approx wpm_adj scaled
for scale in [1, 10, 100, 1000, 0.1, 0.01]:
    diff = np.abs(df['running_time'] - wpm_adj * scale)
    print('scale', scale, 'median diff', np.median(diff))
