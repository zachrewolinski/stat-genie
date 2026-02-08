import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')
children = df['children'].astype(str).str.strip().str.lower()
mask = children.isin(['yes','no'])
sub = df.loc[mask].copy()
sub['children_yes'] = (children[mask] == 'yes').astype(int)
X = sm.add_constant(sub['children_yes'])

nb_model = sm.NegativeBinomial(sub['affairs'], X)
nb_res = nb_model.fit(disp=False)

print('NB2 coef:', nb_res.params['children_yes'])
print('NB2 p-value:', nb_res.pvalues['children_yes'])
print('NB2 rate ratio:', np.exp(nb_res.params['children_yes']))
print('NB2 alpha:', nb_res.params['alpha'])
