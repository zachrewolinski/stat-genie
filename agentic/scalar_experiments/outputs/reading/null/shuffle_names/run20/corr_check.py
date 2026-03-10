import pandas as pd

_df = pd.read_csv('reading.csv')

cols = ['adjusted_running_time','age','gender','running_time','retake_trial','num_words']

for c1 in cols:
    for c2 in cols:
        if c1 < c2:
            corr = _df[c1].corr(_df[c2])
            print(c1, c2, corr)
