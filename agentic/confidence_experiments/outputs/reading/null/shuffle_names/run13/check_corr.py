import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
for col in ['adjusted_running_time','age','gender','retake_trial','num_words']:
    corr = np.corrcoef(df['running_time'], df[col])[0,1]
    print('running_time vs', col, corr)
