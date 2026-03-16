import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']
_df['is_homo'] = (_df['genus'] == 'Homo sapiens').astype(int)

# OLS on amtl_rate
model = smf.ols('amtl_rate ~ is_homo + age + prob_male + C(tooth_class)', data=_df).fit()
coef = model.params['is_homo']
ci_low, ci_high = model.conf_int().loc['is_homo']
pval = model.pvalues['is_homo']

# OLS with genus categorical to check adjusted differences
model_genus = smf.ols('amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit()
# Relevel to Homo sapiens as reference
_df['genus'] = pd.Categorical(_df['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
model_genus_ref = smf.ols('amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit()

means = _df.groupby('genus')['amtl_rate'].mean().to_dict()

print('coef_is_homo', coef)
print('ci_is_homo', (ci_low, ci_high))
print('p_is_homo', pval)
print('r2', model.rsquared)
print('mean_amtl_rate_by_genus', means)

# also means of num_amtl
means_num = _df.groupby('genus')['num_amtl'].mean().to_dict()
print('mean_num_amtl_by_genus', means_num)

# Extract coefficients for genus differences vs Homo
print('genus_coef_vs_homo', model_genus_ref.params.filter(like='C(genus)').to_dict())
print('genus_p_vs_homo', model_genus_ref.pvalues.filter(like='C(genus)').to_dict())

