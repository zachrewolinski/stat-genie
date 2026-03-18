import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


df = pd.read_csv('affairs.csv')

# Prepare variables
outcome = df['feature2']
child = df['feature6'].astype(str).str.lower()
child_ind = (child == 'yes').astype(int)

# Group stats
stats_by_group = {}
for label in ['yes', 'no']:
    vals = outcome[child == label]
    stats_by_group[label] = {
        'n': int(vals.shape[0]),
        'mean': float(vals.mean()),
        'median': float(vals.median()),
        'std': float(vals.std(ddof=1)),
    }

# Welch t-test
vals_yes = outcome[child == 'yes']
vals_no = outcome[child == 'no']
welch_t = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test
mw = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')

# Cohen's d (pooled sd)
n1, n0 = vals_yes.shape[0], vals_no.shape[0]
var1, var0 = vals_yes.var(ddof=1), vals_no.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohen_d = float((vals_yes.mean() - vals_no.mean()) / pooled_sd) if pooled_sd > 0 else np.nan

# 95% CI for difference in means
cm = CompareMeans(DescrStatsW(vals_yes), DescrStatsW(vals_no))
mean_diff = float(vals_yes.mean() - vals_no.mean())
ci_low, ci_high = cm.tconfint_diff(usevar='unequal')

# OLS regression (robust SEs)
df = df.copy()
df['child'] = child_ind

model_simple = smf.ols('feature2 ~ child', data=df).fit(cov_type='HC3')
model_adj = smf.ols('feature2 ~ child + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(cov_type='HC3')

# Logistic regression sensitivity: any positive outcome
# (note: anonymized values may blur original zero threshold)
df['affair_pos'] = (df['feature2'] > 0).astype(int)
model_logit = smf.logit('affair_pos ~ child', data=df).fit(disp=False)

results = {
    'group_stats': stats_by_group,
    'welch_t': {'statistic': float(welch_t.statistic), 'pvalue': float(welch_t.pvalue)},
    'mannwhitney': {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'mean_diff': mean_diff,
    'mean_diff_ci95': [float(ci_low), float(ci_high)],
    'cohen_d': cohen_d,
    'ols_simple': {
        'coef_child': float(model_simple.params['child']),
        'pvalue_child': float(model_simple.pvalues['child']),
        'r2': float(model_simple.rsquared),
    },
    'ols_adj': {
        'coef_child': float(model_adj.params['child']),
        'pvalue_child': float(model_adj.pvalues['child']),
        'r2': float(model_adj.rsquared),
    },
    'logit': {
        'coef_child': float(model_logit.params['child']),
        'pvalue_child': float(model_logit.pvalues['child']),
    }
}

print(json.dumps(results, indent=2))
