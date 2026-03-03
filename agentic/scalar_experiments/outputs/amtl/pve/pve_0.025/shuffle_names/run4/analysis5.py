import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')
# check if any numeric column could be missing count (0<=x<=age)
for col in ['genus','num_amtl','pop','stdev_age']:
    frac = np.mean((df[col] >= 0) & (df[col] <= df['age']))
    print(col, 'fraction between 0 and age', frac)

# for genus compare to age; count how many violate
for col in ['genus','num_amtl','pop']:
    violations = ((df[col] < 0) | (df[col] > df['age'])).sum()
    print(col, 'violations of 0<=x<=age', violations)

