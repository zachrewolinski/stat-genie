import pandas as pd
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print('rows', len(amtl))
print(amtl['num_amtl'].describe())
print('num_amtl min/max', amtl['num_amtl'].min(), amtl['num_amtl'].max())
print('sockets unique', sorted(amtl['sockets'].unique())[:10])
print('genus counts', amtl['genus'].value_counts())
# Check if num_amtl appears as proportion or logit of proportion
# Try to infer underlying counts by checking if values are integers or close when multiplied by sockets
amtl['prop_guess'] = amtl['num_amtl'] / amtl['sockets']
print('prop_guess range', amtl['prop_guess'].min(), amtl['prop_guess'].max())

# Check if num_amtl could be logit of proportion missing
from scipy.special import expit
amtl['inv_logit'] = expit(amtl['num_amtl'])
print('inv_logit range', amtl['inv_logit'].min(), amtl['inv_logit'].max())
# If num_amtl is logit of proportion missing, then count = proportion * sockets should be close to integer
amtl['count_from_logit'] = amtl['inv_logit'] * amtl['sockets']
print('count_from_logit summary')
print(amtl['count_from_logit'].describe())
print('count_from_logit near integer proportion', np.mean(np.isclose(amtl['count_from_logit'] % 1, 0, atol=0.05)))

# Check if num_amtl is maybe logit of proportion of missing among sockets for each record? We'll check correlation with sockets.
print('corr num_amtl with sockets', amtl['num_amtl'].corr(amtl['sockets']))

# Compute mean num_amtl by genus
print(amtl.groupby('genus')['num_amtl'].mean())

