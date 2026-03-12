import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# Check num_amtl and sockets stats
print('num_amtl unique sample', df['num_amtl'].head())
print('sockets unique sample', df['sockets'].head())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())

# Check if num_amtl and sockets are integers
print('num_amtl non-integer count', np.sum(~np.isclose(df['num_amtl'], np.round(df['num_amtl']))))
print('sockets non-integer count', np.sum(~np.isclose(df['sockets'], np.round(df['sockets']))))

# Perhaps num_amtl already standardized; check for possible over 0/1? 
# Try to compute proportion if possible

# check if num_amtl negative
print('num_amtl negative count', (df['num_amtl'] < 0).sum())

# Basic group means of num_amtl and maybe ratio
print(df.groupby('genus')['num_amtl'].agg(['mean','std','min','max','count']))

# If num_amtl likely count but has negatives, maybe it's z-score or centered? Let's inspect variable distribution
print(df['num_amtl'].describe())

# We'll also compute per-socket AMTL if counts are plausible; check if any num_amtl > sockets.
print('num_amtl > sockets count', (df['num_amtl'] > df['sockets']).sum())
print('num_amtl < 0 count', (df['num_amtl'] < 0).sum())

# Build model; perhaps amtl is already standardized continuous; can run linear regression with genus, age, sex, tooth_class.
# We'll try with OLS.

# Convert categorical
for col in ['genus','tooth_class']:
    df[col] = df[col].astype('category')

# Simple OLS
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()
print(model.summary())

# If counts are integers (maybe not), also try binomial with endog as counts / sockets
# Use GLM with binomial and proportion if num_amtl is within [0, sockets]
if (df['num_amtl'].min() >= 0) and (df['num_amtl'] <= df['sockets']).all():
    df['amtl_prop'] = df['num_amtl'] / df['sockets']
    glm_model = smf.glm('amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)', data=df, family=sm.families.Binomial(), freq_weights=df['sockets']).fit()
    print(glm_model.summary())
else:
    print('Binomial model not applicable due to num_amtl range')

