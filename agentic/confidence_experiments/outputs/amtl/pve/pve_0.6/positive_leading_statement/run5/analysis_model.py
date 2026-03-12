import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

df['freq'] = df['num_amtl'] / df['sockets']
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

model1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
model2 = smf.ols('freq ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract key stats

def summarize_model(model, label):
    coef = model.params['is_human']
    se = model.bse['is_human']
    p = model.pvalues['is_human']
    ci_low, ci_high = model.conf_int().loc['is_human']
    print(label)
    print('coef', coef, 'se', se, 'p', p, 'ci', (ci_low, ci_high), 'n', int(model.nobs))

summarize_model(model1, 'OLS num_amtl')
summarize_model(model2, 'OLS freq')

print('Mean freq by genus')
print(df.groupby('genus')['freq'].mean())
print('Mean num_amtl by genus')
print(df.groupby('genus')['num_amtl'].mean())

