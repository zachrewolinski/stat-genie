import pandas as pd
import itertools

_df = pd.read_csv('reading.csv')

num_word_cols = ['retake_trial', 'num_words']
time_cols = ['adjusted_running_time', 'age', 'gender']

for nw in num_word_cols:
    for tc in time_cols:
        # avoid division by zero
        speed = _df[nw] / (_df[tc] / 60000)
        corr = _df['running_time'].corr(speed)
        print(f'num_words={nw}, time={tc}, corr with running_time={corr:.4f}')
