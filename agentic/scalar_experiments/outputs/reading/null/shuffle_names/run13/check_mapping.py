import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

def wpm(words, time_ms):
    return words / (time_ms/60000)

rt = df['running_time']
for words_col in ['retake_trial','num_words']:
    for time_col in ['adjusted_running_time','age','gender']:
        computed = wpm(df[words_col], df[time_col])
        corr = np.corrcoef(rt, computed)[0,1]
        print(words_col, time_col, 'corr', corr)

print('corr adjusted_running_time vs age', np.corrcoef(df['adjusted_running_time'], df['age'])[0,1])
print('corr adjusted_running_time vs gender', np.corrcoef(df['adjusted_running_time'], df['gender'])[0,1])

for col in ['adjusted_running_time','age','gender','running_time','retake_trial','num_words']:
    s = df[col]
    print(col, s.min(), s.max(), s.mean())
