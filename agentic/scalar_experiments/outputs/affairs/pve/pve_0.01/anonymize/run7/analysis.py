import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Variables
outcome = 'feature2'  # affairs frequency
children = 'feature6'

# Basic cleaning
# Ensure expected categories
# drop missing if any
cols_needed = [outcome, children, 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']
df = df.dropna(subset=cols_needed).copy()

# Binary indicator for children
child_yes = df[children].str.lower().map({'yes': 1, 'no': 0})
df['child_yes'] = child_yes

# Basic group stats
stats_by = df.groupby('child_yes')[outcome].agg(['count', 'mean', 'median', 'std'])

# t-test (Welch)
y_yes = df.loc[df['child_yes'] == 1, outcome]
y_no = df.loc[df['child_yes'] == 0, outcome]

ttest = stats.ttest_ind(y_yes, y_no, equal_var=False)

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(y_yes, y_no, alternative='two-sided')
except ValueError:
    mwu = None

# Effect size: Cohen's d (pooled)
mean_yes = y_yes.mean()
mean_no = y_no.mean()
std_yes = y_yes.std(ddof=1)
std_no = y_no.std(ddof=1)

n_yes = len(y_yes)
n_no = len(y_no)
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Any affair indicator
any_affair = (df[outcome] > 0).astype(int)
df['any_affair'] = any_affair

# Proportions
prop_yes = df.loc[df['child_yes'] == 1, 'any_affair'].mean()
prop_no = df.loc[df['child_yes'] == 0, 'any_affair'].mean()

# Two-proportion z-test
count = np.array([
    df.loc[df['child_yes'] == 1, 'any_affair'].sum(),
    df.loc[df['child_yes'] == 0, 'any_affair'].sum()
])
obs = np.array([n_yes, n_no])

try:
    zstat, pval_prop = sm.stats.proportions_ztest(count, obs)
except Exception:
    zstat, pval_prop = np.nan, np.nan

# Regression: OLS with controls
# Encode gender as binary: female=1, male=0
# Use formula with C() for categorical for safety
ols_formula = (
    "feature2 ~ child_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affair
logit_formula = (
    "any_affair ~ child_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# Extract coefficients
ols_coef = ols_model.params.get('child_yes', np.nan)
ols_p = ols_model.pvalues.get('child_yes', np.nan)
ols_ci = ols_model.conf_int().loc['child_yes'].tolist()

logit_coef = logit_model.params.get('child_yes', np.nan)
logit_p = logit_model.pvalues.get('child_yes', np.nan)
logit_ci = logit_model.conf_int().loc['child_yes'].tolist()
# Odds ratio and CI
or_val = float(np.exp(logit_coef)) if np.isfinite(logit_coef) else np.nan
or_ci = [float(np.exp(logit_ci[0])), float(np.exp(logit_ci[1]))] if np.all(np.isfinite(logit_ci)) else [np.nan, np.nan]

results = {
    'n_total': int(len(df)),
    'group_stats': stats_by.to_dict(),
    'welch_ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitneyu': None if mwu is None else {'statistic': float(mwu.statistic), 'pvalue': float(mwu.pvalue)},
    'cohen_d': float(cohen_d),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'prop_any_affair_yes': float(prop_yes),
    'prop_any_affair_no': float(prop_no),
    'prop_ztest': {'z': float(zstat), 'pvalue': float(pval_prop)},
    'ols_child_yes': {'coef': float(ols_coef), 'pvalue': float(ols_p), 'ci_low': float(ols_ci[0]), 'ci_high': float(ols_ci[1])},
    'logit_child_yes': {'coef': float(logit_coef), 'pvalue': float(logit_p), 'or': or_val, 'or_ci_low': float(or_ci[0]), 'or_ci_high': float(or_ci[1])}
}

print(json.dumps(results, indent=2))
