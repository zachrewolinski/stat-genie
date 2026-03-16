import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('amtl.csv')
print(df[['num_amtl','sockets']].corr())
print(df.groupby('genus')['num_amtl'].mean())
print(df.groupby('genus')['num_amtl'].median())
print(df.groupby('genus')['num_amtl'].describe())

# simple OLS with controls
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()
print(model.summary())

# human vs non-human (contrast)

