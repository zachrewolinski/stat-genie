import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# AMTL rate proxy
_df['amtl_rate'] = _df['feature3'] / _df['feature4']

# Fit linear model with genus, tooth class, age, sex
model = smf.ols('amtl_rate ~ C(feature8) + C(feature1) + feature5 + feature7', data=_df).fit(cov_type='HC3')

# Extract genus coefficients (relative to Homo sapiens)
coef = model.params
pvals = model.pvalues

# Adjusted mean AMTL rate by genus (marginalizing over observed covariates)
adj_means = {}
for genus in _df['feature8'].unique():
    tmp = _df.copy()
    tmp['feature8'] = genus
    adj_means[genus] = float(model.predict(tmp).mean())

# ANOVA for overall genus effect
anova = sm.stats.anova_lm(model, typ=2)

print('Adjusted mean AMTL rate by genus:')
for k, v in sorted(adj_means.items()):
    print(f'{k}: {v:.4f}')

print('\nGenus coefficients vs Homo sapiens:')
for genus in ['Pan','Pongo','Papio']:
    term = f'C(feature8)[T.{genus}]'
    if term in coef:
        print(f'{term}: coef={coef[term]:.4f}, p={pvals[term]:.4g}')

print('\nANOVA (type 2):')
print(anova)
