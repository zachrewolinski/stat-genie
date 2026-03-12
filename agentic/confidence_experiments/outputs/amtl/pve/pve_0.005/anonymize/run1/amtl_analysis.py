import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = 'amtl.csv'
df = pd.read_csv(path)

df['missing_prop'] = df['feature3'] / df['feature4']

neg_prop = (df['missing_prop'] < 0).sum()
above_one = (df['missing_prop'] > 1).sum()

print('rows', len(df))
print('neg_prop', neg_prop, 'above_one', above_one)

# Human indicator

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS with robust SE
model = smf.ols('missing_prop ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')
print('OLS human vs non-human')
print(model.summary())

# OLS with genus categories
model_genus = smf.ols('missing_prop ~ C(feature8) + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')
print('OLS full genus')
print(model_genus.summary())

# Descriptive means by genus
means = df.groupby('feature8')['missing_prop'].mean()
print('means', means.to_dict())

# Sensitivity: clip proportion to [0,1] and fit binomial GLM with weights
clipped = df.copy()
clipped['missing_prop_clip'] = clipped['missing_prop'].clip(0, 1)

try:
    glm = smf.glm('missing_prop_clip ~ is_human + feature5 + feature7 + C(feature1)',
                  data=clipped, family=sm.families.Binomial(),
                  freq_weights=clipped['feature4']).fit()
    print('GLM human vs non-human')
    print(glm.summary())
except Exception as e:
    print('GLM failed', e)

try:
    glm_genus = smf.glm('missing_prop_clip ~ C(feature8) + feature5 + feature7 + C(feature1)',
                        data=clipped, family=sm.families.Binomial(),
                        freq_weights=clipped['feature4']).fit()
    print('GLM full genus')
    print(glm_genus.summary())
except Exception as e:
    print('GLM genus failed', e)

