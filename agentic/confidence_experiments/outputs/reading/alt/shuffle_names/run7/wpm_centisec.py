import pandas as pd
import numpy as np

df=pd.read_csv('reading.csv')

# assume age is adjusted time in centiseconds (0.01s)
adj_cs = df['age'].astype(float)
num_words = df['num_words'].astype(float)

# convert centiseconds to seconds: cs * 0.01
wpm = num_words / (adj_cs*0.01/60)

print(wpm.describe())
print('corr wpm vs running_time', np.corrcoef(wpm, df['running_time'])[0,1])
print('corr wpm vs adj_cs', np.corrcoef(wpm, adj_cs)[0,1])
