import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
for words_col in ['retake_trial','num_words']:
    words = df[words_col]
    for time_col in ['adjusted_running_time','age','gender']:
        # assume ms
        wpm = words / (df[time_col]/1000.0) * 60
        corr = np.corrcoef(wpm, df['running_time'])[0,1]
        print(f'words {words_col} time {time_col} corr with running_time {corr}')
