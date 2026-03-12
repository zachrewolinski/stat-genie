import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

num_words = df['retake_trial']  # candidate for num_words
# If retake_trial is num_words, should be 106-383.
print('retake_trial range', num_words.min(), num_words.max())

# test candidate time columns
for col in ['adjusted_running_time','age','gender','running_time']:
    t = df[col].astype(float)
    # avoid zero
    wpm = num_words / (t/60000.0)
    print('\nCandidate time:', col)
    print('wpm median', np.nanmedian(wpm), 'mean', np.nanmean(wpm), 'min', np.nanmin(wpm), 'max', np.nanmax(wpm))
    print('wpm 5-95', np.nanpercentile(wpm, [5,95]))
