import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# candidate word-count columns: retake_trial and num_words (ints)
word_candidates = ['retake_trial','num_words']
# candidate time columns (ms): adjusted_running_time, age, gender
# We'll compute wpm = words / time_ms * 60000

for words_col in word_candidates:
    if words_col not in df.columns:
        continue
    words = df[words_col]
    for time_col in ['adjusted_running_time','age','gender','running_time']:
        if time_col not in df.columns:
            continue
        time = df[time_col]
        wpm = words / time * 60000
        # compare wpm with numeric columns for correlation
        print(f"\nDerived wpm from {words_col}/{time_col}")
        print('wpm summary', wpm.describe())
        for col in numeric_cols:
            if col == time_col:
                continue
            corr = wpm.corr(df[col])
            if np.isfinite(corr) and abs(corr) > 0.7:
                print('  high corr', col, corr)
