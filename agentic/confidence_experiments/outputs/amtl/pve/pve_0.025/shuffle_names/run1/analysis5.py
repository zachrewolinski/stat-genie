import pandas as pd
import numpy as np
from scipy.special import expit, logit

df = pd.read_csv('amtl.csv')

# compute proportion missing using num_amtl and age
prop = df['num_amtl'] / df['age']
# clip to (0,1) for logit
prop_clipped = prop.clip(1e-6, 1-1e-6)
logit_prop = np.log(prop_clipped / (1-prop_clipped))

# correlation between genus and logit_prop
corr = np.corrcoef(df['genus'], logit_prop)[0,1]
print('corr(genus, logit(num_amtl/age))', corr)

# also compare genus with num_amtl and with ratio
print('corr(genus, num_amtl)', df['genus'].corr(df['num_amtl']))
print('corr(genus, num_amtl/age)', df['genus'].corr(prop))

# check if genus approximates num_amtl maybe? compute mae
mae = np.mean(np.abs(df['genus'] - df['num_amtl']))
print('MAE genus vs num_amtl', mae)

# check if num_amtl <= age mostly
print('num_amtl <= age fraction', (df['num_amtl'] <= df['age']).mean())

