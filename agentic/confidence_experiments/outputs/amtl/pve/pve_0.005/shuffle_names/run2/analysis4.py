import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

num_cols = ['genus','age','pop','num_amtl','stdev_age']
print(df[num_cols].corr())

# Check if genus is close to logit of something like num_amtl/age or num_amtl/pop
# We'll compute proportion = num_amtl / age (if age is sockets) etc and see correlation with genus.

for denom in ['age','pop','num_amtl']:
    # avoid divide by zero
    denom_vals = df[denom].replace(0, np.nan)
    prop = df['num_amtl'] / denom_vals
    prop = prop.clip(1e-6, 1-1e-6)
    logit = np.log(prop/(1-prop))
    corr = np.corrcoef(df['genus'], logit)[0,1]
    print('corr genus vs logit(num_amtl/'+denom+')', corr)

# Another guess: genus might be logit of missing proportion using age as sockets and num_amtl as missing; see if genus approximates logit of (genus?)
# Check if expit(genus) correlates with num_amtl/age
from math import exp

logit_vals = df['genus']
expit = 1/(1+np.exp(-logit_vals))
for denom in ['age','pop','num_amtl']:
    denom_vals = df[denom].replace(0, np.nan)
    prop = df['num_amtl']/denom_vals
    corr = np.corrcoef(expit, prop)[0,1]
    print('corr expit(genus) vs num_amtl/'+denom, corr)

