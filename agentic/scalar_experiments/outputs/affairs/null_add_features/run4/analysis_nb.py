import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')
children = df['children'].astype(str).str.strip().str.lower()
mask = children.isin(['yes','no'])
sub = df.loc[mask].copy()
sub['children_yes'] = (children[mask] == 'yes').astype(int)
X = sm.add_constant(sub['children_yes'])

# Negative binomial (NB2) with alpha estimated
nb_model = sm.GLM(sub['affairs'], X, family=sm.families.NegativeBinomial(alpha=1.0))
nb_res = nb_model.fit()

print('NB coef:', nb_res.params['children_yes'])
print('NB p-value:', nb_res.pvalues['children_yes'])
print('NB rate ratio:', np.exp(nb_res.params['children_yes']))
print('Alpha used:', nb_res.scale)
