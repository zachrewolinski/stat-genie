import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print('accept + deny unique', (df['accept'] + df['deny']).unique()[:10])
print('crosstab accept, deny')
print(pd.crosstab(df['accept'], df['deny']))

# determine correlation
print('corr', df['accept'].corr(df['deny']))

