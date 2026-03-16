import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')

# compute proportion using num_amtl / pop (specimen-level)
prop = df['num_amtl'] / df['pop']
# avoid bounds
prop = prop.clip(1e-6, 1-1e-6)
logit = np.log(prop/(1-prop))

corr = np.corrcoef(df['genus'], logit)[0,1]
print('corr genus vs logit(num_amtl/pop)', corr)

corr2 = np.corrcoef(df['genus'], prop)[0,1]
print('corr genus vs prop', corr2)

# check if genus ~ log(num_amtl) maybe
corr3 = np.corrcoef(df['genus'], np.log(df['num_amtl']))[0,1]
print('corr genus vs log(num_amtl)', corr3)

# check if genus ~ log(pop)
print('corr genus vs log(pop)', np.corrcoef(df['genus'], np.log(df['pop']))[0,1])

