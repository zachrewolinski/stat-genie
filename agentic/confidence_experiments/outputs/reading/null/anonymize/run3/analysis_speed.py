import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

def corr_speed(time_col):
    speed = df['feature7'] / (df[time_col] / 60000)
    corr = np.corrcoef(speed, df['feature20'])[0,1]
    return speed.describe(), corr

for col in ['feature4','feature5','feature6']:
    if col in df.columns:
        desc, corr = corr_speed(col)
        print(col, 'corr', corr)
        print(desc[['mean','std','min','25%','50%','75%','max']])
        print()

# Check if feature20 maybe time per word (ms per word)
for col in ['feature4','feature5']:
    time_per_word = df[col] / df['feature7']
    corr = np.corrcoef(time_per_word, df['feature20'])[0,1]
    print('time_per_word from', col, 'corr with feature20', corr)

