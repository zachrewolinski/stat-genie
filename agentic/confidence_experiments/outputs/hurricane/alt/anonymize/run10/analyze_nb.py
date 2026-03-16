import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv('hurricane.csv')
cols = {f'feature{i}': f'f{i}' for i in range(1, 15)}
df = df.rename(columns=cols)

# Negative Binomial GLM
for var in ['f4','f6','f12']:
    formula = f'f8 ~ {var} + f7 + f5 + f13'
    model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial()).fit()
    print('\nNegative Binomial:', formula)
    print(model.summary())

# Also try log_deaths with robust SEs

model_ols = smf.ols('np.log1p(f8) ~ f4 + f7 + f5 + f13', data=df).fit(cov_type='HC3')
print('\nOLS with HC3 robust SE: log_deaths ~ f4 + f7 + f5 + f13')
print(model_ols.summary())

model_ols2 = smf.ols('np.log1p(f8) ~ f6 + f7 + f5 + f13', data=df).fit(cov_type='HC3')
print('\nOLS with HC3 robust SE: log_deaths ~ f6 + f7 + f5 + f13')
print(model_ols2.summary())

