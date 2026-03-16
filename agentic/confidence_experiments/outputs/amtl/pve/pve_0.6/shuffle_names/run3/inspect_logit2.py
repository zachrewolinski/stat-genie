import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')

prop = 1/(1+np.exp(-df['genus']))
missing_est = prop * df['age']

print('missing_est min', missing_est.min(), 'max', missing_est.max())
print('missing_est <= age share', (missing_est <= df['age']).mean())
print('missing_est mean', missing_est.mean())

# check correlation between missing_est and num_amtl
print('corr missing_est vs num_amtl', np.corrcoef(missing_est, df['num_amtl'])[0,1])

