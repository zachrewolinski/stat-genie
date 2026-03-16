import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

# candidate time columns
candidates = ['adjusted_running_time','age','gender']
# candidate word count columns
word_candidates = ['num_words','retake_trial']

for time_col in candidates:
    for word_col in word_candidates:
        # compute wpm
        wpm = df[word_col] / (df[time_col] / 60000.0)
        # correlation with running_time column
        corr = wpm.corr(df['running_time'])
        print(f"wpm({word_col}/{time_col}) corr with running_time: {corr:.3f}")

# correlations between running_time and time columns
for time_col in candidates:
    corr = df['running_time'].corr(df[time_col])
    print(f"corr running_time with {time_col}: {corr:.3f}")

# Check running_time vs computed wpm for best match
