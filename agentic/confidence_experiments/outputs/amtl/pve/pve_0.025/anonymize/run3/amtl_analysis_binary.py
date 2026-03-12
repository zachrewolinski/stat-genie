import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

df['feature1'] = df['feature1'].astype('category')

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')

coef = model.params['is_human']
pval = model.pvalues['is_human']
ci_low, ci_high = model.conf_int().loc['is_human']

print('is_human coef:', coef)
print('p_value:', pval)
print('95% CI:', ci_low, ci_high)
