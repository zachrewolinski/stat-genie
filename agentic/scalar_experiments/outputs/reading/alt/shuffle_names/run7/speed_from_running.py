import pandas as pd
import numpy as np


df=pd.read_csv('reading.csv')

running = df['running_time']
num_words = df['num_words']

wpm = num_words / (running/60)
print(wpm.describe())

# correlation with adjusted time (age) and total time (adjusted_running_time)
print('corr wpm vs age', np.corrcoef(wpm, df['age'])[0,1])
print('corr wpm vs adjusted_running_time', np.corrcoef(wpm, df['adjusted_running_time'])[0,1])

# check correlation between running_time and adjusted time
print('corr running_time vs age', np.corrcoef(running, df['age'])[0,1])

