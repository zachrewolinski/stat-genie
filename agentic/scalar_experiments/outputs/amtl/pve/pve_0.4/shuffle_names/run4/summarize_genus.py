import pandas as pd

# Load data

df = pd.read_csv('amtl.csv')

print('genus min', df['genus'].min(), 'max', df['genus'].max())
print('genus negative proportion', (df['genus'] < 0).mean())
print(df.groupby('sockets')['genus'].agg(['min','max','mean']))

# Check if genus maybe logit of proportion? compute proportion = expit(genus)
import numpy as np
from scipy.special import expit

df['p'] = expit(df['genus'])
print('p range', df['p'].min(), df['p'].max())
# if p*age roughly equals something? check correlation with num_amtl maybe?
print('corr p*age with num_amtl', (df['p'] * df['age']).corr(df['num_amtl']))
print('corr p*age with age', (df['p'] * df['age']).corr(df['age']))
