import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df['genus'].value_counts())
print(df['tooth_class'].value_counts())
print(df[['num_amtl','sockets']].describe())

# check if num_amtl maybe already standardized? ensure sockets integer
# compute proportion

# if num_amtl and sockets are counts, maybe use binomial with num_amtl successes
# But num_amtl can be negative, so not counts. Perhaps it is transformed (standardized) or residualized?
# Let's inspect min/max within each genus.
print(df.groupby('genus')['num_amtl'].agg(['min','max','mean','std']))

# Check if num_amtl is integer?
print('num_amtl integer proportion:', np.mean(np.isclose(df['num_amtl'], np.round(df['num_amtl']))))

# Maybe num_amtl is logit of proportion? Let's compute amtl proportion if num_amtl is counts? We'll check if num_amtl within [0,sockets].
within = (df['num_amtl']>=0) & (df['num_amtl']<=df['sockets'])
print('within 0..sockets proportion', within.mean())

# Perhaps num_amtl is z-scored or scaled variable. Could attempt to infer raw counts by z-score? Hmm.

