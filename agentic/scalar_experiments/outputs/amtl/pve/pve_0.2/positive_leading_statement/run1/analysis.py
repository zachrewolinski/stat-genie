import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('sockets unique', sorted(df['sockets'].unique())[:10])

# check if num_amtl integer-like
num_amtl_nonint = ((df['num_amtl'] % 1)!=0).mean()
print('fraction non-integer num_amtl', num_amtl_nonint)

# maybe num_amtl is standardized; check if there is column with missing teeth? no.
# maybe num_amtl is log-odds of proportion? We'll check relation with sockets.

# check if any values negative or > sockets
print('num_amtl min', df['num_amtl'].min(), 'max', df['num_amtl'].max())
print('sockets min', df['sockets'].min(), 'max', df['sockets'].max())

# inspect per genus mean
print(df.groupby('genus')['num_amtl'].mean())
