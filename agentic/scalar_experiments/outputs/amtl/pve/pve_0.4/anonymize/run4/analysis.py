import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create human indicator
df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Fit OLS with cluster-robust SE by specimen id (feature2)
formula = 'feature3 ~ is_human + feature5 + feature7 + C(feature1)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

# Also fit with genus categories to inspect differences
model_genus = smf.ols('feature3 ~ C(feature8) + feature5 + feature7 + C(feature1)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['feature2']}
)

# Compute adjusted means difference using is_human coefficient
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Basic descriptive stats
mean_human = df.loc[df['is_human'] == 1, 'feature3'].mean()
mean_nonhuman = df.loc[df['is_human'] == 0, 'feature3'].mean()

print('OLS human effect (cluster-robust)')
print('coef', coef)
print('se', se)
print('pval', pval)
print('mean_human', mean_human)
print('mean_nonhuman', mean_nonhuman)

print('\nModel summary (human vs nonhuman)')
print(model.summary())

print('\nModel summary (genus categories)')
print(model_genus.summary())

# Save key stats to csv for easy reference
out = pd.DataFrame({
    'coef_is_human': [coef],
    'se_is_human': [se],
    'pval_is_human': [pval],
    'mean_human': [mean_human],
    'mean_nonhuman': [mean_nonhuman],
})
out.to_csv('analysis_key_stats.csv', index=False)
