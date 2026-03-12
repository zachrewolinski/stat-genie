import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
num_cols = ['genus','age','pop','num_amtl','stdev_age']
print(df[num_cols].corr())

# explore counts by genus (tooth_class) for each numeric column
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\n', col)
    print(df.groupby('tooth_class')[col].mean())
