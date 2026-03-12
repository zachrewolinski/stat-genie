import pandas as pd
import numpy as np


df=pd.read_csv('reading.csv')

# map: total_time_ms = adjusted_running_time, adjusted_time_ms = age, scrolling_time_ms = gender

adjusted_ms = df['age'].astype(float)
num_words = df['num_words'].astype(float)

# compute words per minute using adjusted time
wpm = num_words / (adjusted_ms/1000/60)

print('wpm stats', wpm.describe())

# compare wpm to running_time and uuid
for col in ['running_time','uuid','retake_trial','Flesch_Kincaid']:
    corr = np.corrcoef(wpm, df[col].astype(float))[0,1]
    print('corr wpm vs', col, corr)

# check if running_time equals wpm or 1/wpm etc
rt = df['running_time']
print('median ratio wpm/rt', (wpm/rt).median())
print('median ratio rt/wpm', (rt/wpm).median())

# check if running_time equals adjusted_ms/num_words or similar
ratio = adjusted_ms/num_words
print('median ratio running_time / (adjusted_ms/num_words)', (rt/ratio).median())
print('median ratio (adjusted_ms/num_words) / running_time', (ratio/rt).median())

# check if running_time correlates with adjusted_ms
print('corr running_time vs adjusted_ms', np.corrcoef(rt, adjusted_ms)[0,1])

