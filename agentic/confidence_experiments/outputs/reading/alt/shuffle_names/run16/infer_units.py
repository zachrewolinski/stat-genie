import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
num_words = df['num_words']

# candidate time columns (actual time): age (adjusted), adjusted_running_time (total)
for time_col in ['age','adjusted_running_time']:
    t = df[time_col]
    print('\nTime col', time_col)
    # unit factors: if t is ms, sec = t/1000; if centisec, sec = t/100; if decisecond t/10
    for unit, denom in [('ms',1000),('cs',100),('ds',10),('sec',1)]:
        wpm = num_words * 60 / (t/denom)
        diff = (df['running_time'] - wpm).abs().median()
        corr = np.corrcoef(wpm, df['running_time'])[0,1]
        print(unit, 'median diff', diff, 'corr', corr, 'wpm mean', wpm.mean())
