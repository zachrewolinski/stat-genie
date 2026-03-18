import pandas as pd
import numpy as np

df=pd.read_csv('affairs.csv')
col=df['education']
print('min', col.min(), 'max', col.max())
print('quantiles', col.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

# count how many > 1000, > 2000, >9000
for thr in [100,500,1000,2000,5000,9000]:
    print('> ',thr, (col>thr).sum())

# show smallest 20 and largest 20
print('smallest 20:', col.sort_values().head(20).tolist())
print('largest 20:', col.sort_values().tail(20).tolist())
