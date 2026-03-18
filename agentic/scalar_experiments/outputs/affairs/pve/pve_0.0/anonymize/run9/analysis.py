import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv("affairs.csv")

children_col = "feature6"  # yes/no
affairs_col = "feature2"   # frequency-coded numeric

children = df[children_col].astype(str).str.strip().str.lower()

summary = df.groupby(children_col)[affairs_col].agg(['count', 'mean', 'median', 'std'])

x_yes = df.loc[children == 'yes', affairs_col]
x_no = df.loc[children == 'no', affairs_col]

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(x_yes, x_no, alternative='two-sided')
    mwu_u = float(mwu.statistic)
    mwu_p = float(mwu.pvalue)
except Exception:
    mwu_u = np.nan
    mwu_p = np.nan

# Welch t-test
tt = stats.ttest_ind(x_yes, x_no, equal_var=False, nan_policy='omit')

tt_t = float(tt.statistic)
tt_p = float(tt.pvalue)

# Binary: any affair (>0)
any_affair = (df[affairs_col] > 0).astype(int)
ct = pd.crosstab(children, any_affair)

# Chi-square test
chi2, chi_p, chi_dof, chi_exp = stats.chi2_contingency(ct)

# Logistic regression for any affair ~ children (no covariates)
# Encode children yes=1, no=0
child_bin = (children == 'yes').astype(int)
logit = sm.Logit(any_affair, sm.add_constant(child_bin))
try:
    logit_res = logit.fit(disp=False)
    logit_coef = float(logit_res.params[1])
    logit_p = float(logit_res.pvalues[1])
    logit_or = float(np.exp(logit_coef))
except Exception:
    logit_coef = np.nan
    logit_p = np.nan
    logit_or = np.nan

results = {
    "summary": summary.to_dict(),
    "mwu_u": mwu_u,
    "mwu_p": mwu_p,
    "tt_t": tt_t,
    "tt_p": tt_p,
    "chi2": float(chi2),
    "chi_p": float(chi_p),
    "logit_coef": logit_coef,
    "logit_p": logit_p,
    "logit_or": logit_or,
    "ct": ct.to_dict()
}

print(json.dumps(results, indent=2))
