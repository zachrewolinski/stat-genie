import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# compute expected missing if genus is logit
p = 1/(1+np.exp(-df['genus']))
exp_missing = p * df['age']

# see correlation with num_amtl
corr = np.corrcoef(exp_missing, df['num_amtl'])[0,1]
print('corr(exp_missing, num_amtl)=', corr)
print('exp_missing range', exp_missing.min(), exp_missing.max(), exp_missing.mean())
