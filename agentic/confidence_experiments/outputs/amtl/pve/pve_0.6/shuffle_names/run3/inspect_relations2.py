import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')

ratio = df['num_amtl']/df['age']
print('corr genus vs num_amtl/age', np.corrcoef(df['genus'], ratio)[0,1])

# maybe genus is log(num_amtl/age)
ratio = ratio.clip(1e-6)
print('corr genus vs log(num_amtl/age)', np.corrcoef(df['genus'], np.log(ratio))[0,1])

# maybe genus is num_amtl/age - 1?
print('corr genus vs num_amtl', np.corrcoef(df['genus'], df['num_amtl'])[0,1])
print('corr genus vs age', np.corrcoef(df['genus'], df['age'])[0,1])

