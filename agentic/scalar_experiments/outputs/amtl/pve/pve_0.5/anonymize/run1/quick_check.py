import pandas as pd
import numpy as np

path='amtl.csv'
df=pd.read_csv(path)
print('rows', len(df))
print(df.head())
print('feature3 min max', df['feature3'].min(), df['feature3'].max())
print('feature4 min max', df['feature4'].min(), df['feature4'].max())
print('feature3 > feature4 count', (df['feature3']>df['feature4']).sum())
print('feature3 < 0 count', (df['feature3']<0).sum())
print('feature3 <= feature4? proportion', (df['feature3']<=df['feature4']).mean())

# Check if feature3 looks like logit of proportion between 0 and 1
# derive p = expit(feature3)
from scipy.special import expit
p = expit(df['feature3'])
print('expit(feature3) range', p.min(), p.max())
# check if feature3 maybe log of count? then exp(feature3) range
print('exp(feature3) range', np.exp(df['feature3']).min(), np.exp(df['feature3']).max())

# check correlation feature3 with feature4
print('corr feature3 with feature4', df['feature3'].corr(df['feature4']))

# check per genus mean feature3 and proportion expit
print(df.groupby('feature8')['feature3'].mean())
print(df.groupby('feature8')['feature3'].std())
print(df.groupby('feature8')['feature4'].mean())

# Check if feature3 is proportional to feature4 (linear regression)
import statsmodels.api as sm
X=sm.add_constant(df['feature4'])
model=sm.OLS(df['feature3'], X).fit()
print(model.params)
print(model.rsquared)
