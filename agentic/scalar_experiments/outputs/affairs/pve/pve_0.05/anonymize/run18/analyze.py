import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = 'affairs.csv'

df = pd.read_csv(DATA_PATH)

# Ensure expected columns

affairs = df['feature2']
children = df['feature6']

# Group stats
stats_by_child = df.groupby('feature6')['feature2'].agg(['count','mean','median','std']).to_dict(orient='index')

# two-sample t-test (Welch)
child_yes = df.loc[df['feature6'] == 'yes', 'feature2']
child_no = df.loc[df['feature6'] == 'no', 'feature2']

# t-test
welch_t = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mw = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')

# Cohen's d (using pooled SD)
mean_yes = child_yes.mean()
mean_no = child_no.mean()
var_yes = child_yes.var(ddof=1)
var_no = child_no.var(ddof=1)

def cohen_d(m1, m2, v1, v2, n1, n2):
    # pooled sd (unbiased)
    pooled_var = ((n1-1)*v1 + (n2-1)*v2) / (n1 + n2 - 2)
    return (m1 - m2) / np.sqrt(pooled_var)

d = cohen_d(mean_yes, mean_no, var_yes, var_no, len(child_yes), len(child_no))

# Regression: OLS on raw affairs
# Using categorical for gender and children
ols_formula = 'feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
ols_model = smf.ols(ols_formula, data=df).fit()

# Logit for any affair
# Add small constant in case of perfect separation? We'll try and catch

df['any_affair'] = (df['feature2'] > 0).astype(int)
logit_formula = 'any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
try:
    logit_model = smf.logit(logit_formula, data=df).fit(disp=0)
    logit_params = logit_model.params
    logit_pvalues = logit_model.pvalues
    logit_ok = True
except Exception as e:
    logit_ok = False
    logit_params = None
    logit_pvalues = None
    logit_error = str(e)

# Also run OLS on log1p to reduce skew
log_ols_formula = 'np.log1p(feature2) ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
log_ols_model = smf.ols(log_ols_formula, data=df).fit()

output = {
    'group_stats': stats_by_child,
    'welch_t': {'statistic': float(welch_t.statistic), 'pvalue': float(welch_t.pvalue)},
    'mannwhitney': {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'cohens_d_yes_minus_no': float(d),
    'ols_children_coef': {
        'coef': float(ols_model.params.get('C(feature6)[T.yes]', np.nan)),
        'pvalue': float(ols_model.pvalues.get('C(feature6)[T.yes]', np.nan))
    },
    'log_ols_children_coef': {
        'coef': float(log_ols_model.params.get('C(feature6)[T.yes]', np.nan)),
        'pvalue': float(log_ols_model.pvalues.get('C(feature6)[T.yes]', np.nan))
    },
    'logit_ok': logit_ok,
}
if logit_ok:
    output['logit_children_coef'] = {
        'coef': float(logit_params.get('C(feature6)[T.yes]', np.nan)),
        'pvalue': float(logit_pvalues.get('C(feature6)[T.yes]', np.nan))
    }
else:
    output['logit_error'] = logit_error

with open('analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
