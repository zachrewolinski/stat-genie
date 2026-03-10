import pandas as pd
import numpy as np


df = pd.read_csv('mortgage.csv')

# check complement between deny and self_employed
print('unique of deny + self_employed:', np.unique(df['deny'] + df['self_employed']))
print('mismatch count:', (df['deny'] + df['self_employed'] != 1).sum())

# check complement between deny and accept (original names)
print('unique of deny + accept:', np.unique(df['deny'] + df['accept']))
print('mismatch count deny+accept !=1:', (df['deny'] + df['accept'] != 1).sum())
