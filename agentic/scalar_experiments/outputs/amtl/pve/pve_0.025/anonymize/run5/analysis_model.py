import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'amtl.csv'
df = pd.read_csv(csv_path)

# Create human indicator
# feature8 contains genus labels

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Compute rate (missing per observable sockets)
# feature4 is number of observable sockets
# avoid division by zero

df = df[df['feature4'] > 0].copy()
df['missing_rate'] = df['feature3'] / df['feature4']

# Summary stats
summary = df.groupby('feature8')['missing_rate'].agg(['mean','std','count'])
print('Missing rate by genus')
print(summary)

# OLS on missing rate with weights
# use categorical tooth class (feature1)

model_rate = smf.wls('missing_rate ~ is_human + feature5 + feature7 + C(feature1)', data=df, weights=df['feature4'])
res_rate = model_rate.fit(cov_type='HC3')
print('\nWLS on missing_rate (weights=feature4)')
print(res_rate.summary())

# OLS on counts with feature4 as covariate
model_count = smf.ols('feature3 ~ is_human + feature4 + feature5 + feature7 + C(feature1)', data=df)
res_count = model_count.fit(cov_type='HC3')
print('\nOLS on feature3 with feature4 covariate')
print(res_count.summary())

# Optional: GLM binomial with clipped counts
# create pseudo-counts
clipped = df.copy()
clipped['missing_clipped'] = clipped['feature3'].round().astype(int)
# clip to [0, feature4]
clipped['missing_clipped'] = clipped['missing_clipped'].clip(lower=0)
clipped['missing_clipped'] = clipped[['missing_clipped','feature4']].min(axis=1)

clipped['present_clipped'] = clipped['feature4'] - clipped['missing_clipped']
# Some may be negative if feature4 < missing_clipped (after min it shouldn't)

endog = np.vstack([clipped['missing_clipped'], clipped['present_clipped']]).T

glm_binom = sm.GLM(endog, sm.add_constant(pd.get_dummies(clipped[['is_human','feature5','feature7','feature1']], drop_first=True)), family=sm.families.Binomial())
res_glm = glm_binom.fit()
print('\nGLM binomial on clipped counts')
print(res_glm.summary())

# Compute effect of human from model_rate
coef_rate = res_rate.params['is_human']
se_rate = res_rate.bse['is_human']
print('\nRate model is_human coef', coef_rate, 'SE', se_rate, 'p', res_rate.pvalues['is_human'])

coef_count = res_count.params['is_human']
print('Count model is_human coef', coef_count, 'p', res_count.pvalues['is_human'])

# for glm
# locate is_human coef
if 'is_human' in res_glm.params.index:
    print('GLM is_human coef', res_glm.params['is_human'], 'p', res_glm.pvalues['is_human'])

