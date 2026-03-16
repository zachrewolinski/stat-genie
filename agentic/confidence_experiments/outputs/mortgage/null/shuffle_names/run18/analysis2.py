import pandas as pd
import numpy as np


df = pd.read_csv('mortgage.csv')

# Check relationship between accept and deny
ct = pd.crosstab(df['accept'], df['deny'])
print('crosstab accept vs deny')
print(ct)
print('accept unique:', df['accept'].unique())
print('deny unique:', df['deny'].unique())
print('accept+deny unique:', np.unique(df['accept'] + df['deny']))

# mean rates
print('mean accept', df['accept'].mean())
print('mean deny', df['deny'].mean())

# female distribution
print('female mean', df['female'].mean())

# cross female vs deny
print('female vs deny')
print(pd.crosstab(df['female'], df['deny']))

# cross female vs accept
print('female vs accept')
print(pd.crosstab(df['female'], df['accept']))
