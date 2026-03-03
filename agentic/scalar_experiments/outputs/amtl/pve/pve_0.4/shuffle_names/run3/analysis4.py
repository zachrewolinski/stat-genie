import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('corr genus num_amtl', df['genus'].corr(df['num_amtl']))
print('corr genus sqrt(num_amtl)', df['genus'].corr(np.sqrt(df['num_amtl'])))
print('corr genus log(num_amtl)', df['genus'].corr(np.log(df['num_amtl'])))

# check if genus maybe standardized num_amtl
z = (df['num_amtl'] - df['num_amtl'].mean())/df['num_amtl'].std()
print('corr genus z(num_amtl)', df['genus'].corr(z))
print('mean diff genus - z', (df['genus']-z).abs().mean())

