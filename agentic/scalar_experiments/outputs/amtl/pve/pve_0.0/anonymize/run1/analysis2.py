import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS with clustered SEs by specimen ID
model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['feature2']}
)

print(model.summary())

# Extract coefficient and p-value for is_human
coef = model.params['is_human']
se = model.bse['is_human']
p = model.pvalues['is_human']

print('is_human coef', coef, 'se', se, 'p', p)

# Also run model with genus categories (Homo sapiens baseline?)
model2 = smf.ols('feature3 ~ C(feature8) + feature5 + feature7 + C(feature1)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['feature2']}
)
print(model2.summary())

# Compute adjusted mean difference between human and non-human using model2 predictions

# Create reference rows for each genus with mean covariates and each tooth class? We'll compute average predictions by genus using observed covariate distribution.

preds = model2.predict(df)
# add residual? actual predicted values.
# compute mean predicted by genus
mean_pred = df.assign(pred=preds).groupby('feature8')['pred'].mean()
print('mean predicted by genus')
print(mean_pred)

# Compute difference between human and non-human average predicted
non_human_mean = df.loc[df['feature8'] != 'Homo sapiens'].assign(pred=preds).groupby('feature8')['pred'].mean().mean()
print('human mean pred', mean_pred['Homo sapiens'])
print('non-human mean pred avg', non_human_mean)

