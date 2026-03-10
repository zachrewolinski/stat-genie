import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print('deny == 1 - self_employed', (df['deny'] == (1 - df['self_employed'])).mean())
print('deny == self_employed', (df['deny'] == df['self_employed']).mean())
print('deny == 1 - accept', (df['deny'] == (1 - df['accept'])).mean())
print('accept == self_employed', (df['accept'] == df['self_employed']).mean())
