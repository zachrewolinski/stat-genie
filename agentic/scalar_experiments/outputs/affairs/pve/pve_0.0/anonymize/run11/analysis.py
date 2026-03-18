import json
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Column mapping from metadata
col_affairs = 'feature2'   # frequency of extramarital affairs (numeric scale)
col_children = 'feature6'  # yes/no children
col_gender = 'feature3'
col_age = 'feature4'
col_years_married = 'feature5'
col_relig = 'feature7'
col_edu = 'feature8'
col_occ = 'feature9'
col_marriage = 'feature10'

# Coerce numeric columns
for c in [col_affairs, col_age, col_years_married, col_relig, col_edu, col_occ, col_marriage]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

req = [col_affairs, col_children, col_gender, col_age, col_years_married, col_relig, col_edu, col_occ, col_marriage]
base = df.dropna(subset=req).copy()

# Binary indicators
base['child_yes'] = (base[col_children].astype(str).str.lower() == 'yes').astype(int)
base['male'] = (base[col_gender].astype(str).str.lower() == 'male').astype(int)

# Group stats
stats_by_child = base.groupby('child_yes')[col_affairs].agg(['count','mean','median','std'])

# Welch t-test on mean affairs
grp_yes = base.loc[base['child_yes'] == 1, col_affairs]
grp_no = base.loc[base['child_yes'] == 0, col_affairs]
ttest = stats.ttest_ind(grp_yes, grp_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric, ordinal-friendly)
try:
    mwu = stats.mannwhitneyu(grp_yes, grp_no, alternative='two-sided')
except Exception:
    mwu = None

# Effect size (Cohen's d)
mean_yes = grp_yes.mean()
mean_no = grp_no.mean()
var_yes = grp_yes.var(ddof=1)
var_no = grp_no.var(ddof=1)
n_yes = grp_yes.shape[0]
n_no = grp_no.shape[0]
pooled_sd = (((n_yes - 1) * var_yes + (n_no - 1) * var_no) / (n_yes + n_no - 2)) ** 0.5
cohen_d = (mean_yes - mean_no) / pooled_sd

# Regression with controls (robust SE)
formula = (
    'feature2 ~ child_yes + male + feature4 + feature5 + feature7 + '
    'feature8 + feature9 + feature10'
)
ols = smf.ols(formula, data=base).fit(cov_type='HC3')

output = {
    'n_total': int(base.shape[0]),
    'stats_by_child': stats_by_child.to_dict(),
    'ttest': {'stat': float(ttest.statistic), 'p': float(ttest.pvalue)},
    'mwu': None if mwu is None else {'stat': float(mwu.statistic), 'p': float(mwu.pvalue)},
    'cohen_d': float(cohen_d),
    'ols': {
        'coef_child': float(ols.params['child_yes']),
        'p_child': float(ols.pvalues['child_yes'])
    }
}

print(json.dumps(output, indent=2, sort_keys=True))
