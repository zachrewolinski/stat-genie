import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

amtl = pd.read_csv('amtl.csv')

print('num_amtl mean by genus')
print(amtl.groupby('genus')['num_amtl'].mean())
print('num_amtl std by genus')
print(amtl.groupby('genus')['num_amtl'].std())

print('corr num_amtl vs sockets', amtl['num_amtl'].corr(amtl['sockets']))

# linear regression
amtl['genus'] = amtl['genus'].astype('category')
amtl['tooth_class'] = amtl['tooth_class'].astype('category')

# Use Homo sapiens as reference
amtl['genus'] = amtl['genus'].cat.reorder_categories(['Homo sapiens','Pan','Pongo','Papio'], ordered=False)

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=amtl).fit()
print(model.summary())

# coefficient comparisons: Homo sapiens vs others (others negative indicates lower than humans)
# Evaluate difference in least squares means? But in OLS with Homo reference, coefficients are other - Homo.

