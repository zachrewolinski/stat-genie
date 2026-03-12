import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# check if adjusted_running_time approx age + gender
calc = df['age'] + df['gender']
max_diff = np.nanmax(np.abs(calc - df['adjusted_running_time']))
print('max diff age+gender vs adjusted_running_time', max_diff)
print('corr', np.corrcoef(calc, df['adjusted_running_time'])[0,1])
