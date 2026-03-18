import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Basic prep
# Ensure feature6 is lowercase yes/no
if df['feature6'].dtype != object:
    df['feature6'] = df['feature6'].astype(str)

# Outcome
outcome = df['feature2']

# Group stats
summary = df.groupby('feature6')['feature2'].agg(['count', 'mean', 'median', 'std']).to_dict()

# Welch t-test
yes = df.loc[df['feature6'] == 'yes', 'feature2']
no = df.loc[df['feature6'] == 'no', 'feature2']

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception:
    u_stat, u_p = (np.nan, np.nan)

# Cohen's d
n1, n2 = len(yes), len(no)
mean1, mean2 = np.mean(yes), np.mean(no)
var1, var2 = np.var(yes, ddof=1), np.var(no, ddof=1)
pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohen_d = (mean1 - mean2) / pooled if pooled != 0 else np.nan

# Regression with controls
# Create binary indicator for children for stable parameter naming
df['children_yes'] = (df['feature6'] == 'yes').astype(int)
# We'll use formula with C(feature3) and include other numeric covariates
formula = "feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
ols = smf.ols(formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affair (>0)
df['affair_any'] = (df['feature2'] > 0).astype(int)
logit = smf.logit(
    "affair_any ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df
).fit(disp=False)

logit_params = logit.params
logit_se = logit.bse
logit_p = logit.pvalues

result = {
    "group_summary": summary,
    "t_test": {"t": float(t_stat), "p": float(t_p)},
    "mannwhitney": {"u": float(u_stat), "p": float(u_p)},
    "cohen_d_yes_minus_no": float(cohen_d),
    "ols_children_coef": float(ols.params["children_yes"]),
    "ols_children_p": float(ols.pvalues["children_yes"]),
    "ols_children_ci": [float(x) for x in ols.conf_int().loc["children_yes"].tolist()],
    "logit_children_coef": float(logit_params["children_yes"]),
    "logit_children_p": float(logit_p["children_yes"]),
    "logit_children_or": float(np.exp(logit_params["children_yes"])),
    "logit_children_or_ci": [float(np.exp(x)) for x in logit.conf_int().loc["children_yes"].tolist()],
    "counts": {"yes": int(n1), "no": int(n2)},
}

print(json.dumps(result, indent=2))
