import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')
skin = df[['feature18','feature19']].mean(axis=1)
mask = skin.notna() & df['feature9'].notna() & df['feature16'].notna() & (df['feature9'] > 0)
use = df.loc[mask].copy()
use['skin_mean'] = skin[mask]

X = sm.add_constant(use['skin_mean'])
model = sm.GLM(use['feature16'], X, family=sm.families.Poisson(), offset=np.log(use['feature9']))
res = model.fit(cov_type='HC1')

beta = res.params['skin_mean']
se = res.bse['skin_mean']
pval = res.pvalues['skin_mean']
rate_ratio = float(np.exp(beta * (0.75 - 0.25)))

print({'n': len(use), 'beta': float(beta), 'se': float(se), 'p': float(pval), 'rate_ratio_075_025': rate_ratio})
