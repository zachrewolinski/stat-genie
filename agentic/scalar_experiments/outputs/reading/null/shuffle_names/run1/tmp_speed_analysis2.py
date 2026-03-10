import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate words columns: retake_trial, num_words
# candidate reading time columns: age (adjusted running time), adjusted_running_time (running time), gender (scrolling time)

candidates = []
for words_col in ['retake_trial','num_words']:
    if words_col not in df.columns:
        continue
    for time_col in ['age','adjusted_running_time','gender']:
        if time_col not in df.columns:
            continue
        speed = df[words_col] * 60000 / df[time_col]
        candidates.append((words_col, time_col, speed))

for words_col, time_col, speed in candidates:
    corr = df['running_time'].corr(speed)
    print(f"corr running_time vs speed from {words_col}/{time_col}: {corr:.3f}")
    print(speed.describe())
    print()

