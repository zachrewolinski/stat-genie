import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Binary indicator for modern humans
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with robust SEs
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
# 95% CI
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
pval = model.pvalues['is_human']

# Compute adjusted means for humans vs non-humans at mean covariates
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# use overall distribution of tooth_class for marginal mean
# get predicted mean for each row but set is_human; then average
_df_pred_base = _df.copy()
_df_pred_base['age'] = mean_age
_df_pred_base['prob_male'] = mean_prob_male

for cls in _df['tooth_class'].unique():
    pass

_df_pred_human = _df_pred_base.copy()
_df_pred_human['is_human'] = 1
_df_pred_non = _df_pred_base.copy()
_df_pred_non['is_human'] = 0

pred_human = model.predict(_df_pred_human).mean()
pred_non = model.predict(_df_pred_non).mean()

# Simple effect size (standardized by outcome SD)
std_y = _df['num_amtl'].std(ddof=1)
std_effect = coef / std_y

print('coef_is_human', coef)
print('se', se)
print('pval', pval)
print('ci', ci_low, ci_high)
print('adj_mean_human', pred_human)
print('adj_mean_non', pred_non)
print('std_effect', std_effect)

# Save key results
pd.Series({
    'coef_is_human': coef,
    'se_is_human': se,
    'pval_is_human': pval,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'adj_mean_human': pred_human,
    'adj_mean_nonhuman': pred_non,
    'std_effect': std_effect,
}).to_json('analysis_results.json')
