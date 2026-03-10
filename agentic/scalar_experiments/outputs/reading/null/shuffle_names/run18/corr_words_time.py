import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

word_cols=['retake_trial','num_words']
time_cols=['adjusted_running_time','age','gender','running_time']

for w in word_cols:
    for t in time_cols:
        corr = df[w].corr(df[t])
        print(f"corr {w} vs {t}: {corr:.3f}")
