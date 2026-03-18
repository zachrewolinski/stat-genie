import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "affairs.csv"
info_path = "info.json"

df = pd.read_csv(csv_path)

# Map columns
# feature2 = affairs frequency, feature6 = children yes/no

# Clean
# ensure children is binary
children = df["feature6"].astype(str).str.lower().str.strip()

# Some datasets may have unexpected values; keep only yes/no
mask = children.isin(["yes", "no"])

df = df.loc[mask].copy()
children = children.loc[mask]

affairs = pd.to_numeric(df["feature2"], errors="coerce")

# Remove missing
valid = affairs.notna()

df = df.loc[valid].copy()
children = children.loc[valid]
affairs = affairs.loc[valid]

# groups
arr_yes = affairs[children == "yes"]
arr_no = affairs[children == "no"]

summary = {
    "n_total": int(len(df)),
    "n_children_yes": int(len(arr_yes)),
    "n_children_no": int(len(arr_no)),
    "mean_affairs_yes": float(arr_yes.mean()),
    "mean_affairs_no": float(arr_no.mean()),
    "median_affairs_yes": float(arr_yes.median()),
    "median_affairs_no": float(arr_no.median()),
    "prop_any_affair_yes": float((arr_yes > 0).mean()),
    "prop_any_affair_no": float((arr_no > 0).mean()),
}

# Welch t-test
try:
    t_stat, t_p = stats.ttest_ind(arr_yes, arr_no, equal_var=False, nan_policy="omit")
except Exception:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U
try:
    u_stat, u_p = stats.mannwhitneyu(arr_yes, arr_no, alternative="two-sided")
except Exception:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d
# Use pooled standard deviation (unbiased) for effect size
n1, n2 = len(arr_yes), len(arr_no)
var1, var2 = arr_yes.var(ddof=1), arr_no.var(ddof=1)
if n1 > 1 and n2 > 1:
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = (arr_yes.mean() - arr_no.mean()) / pooled_sd if pooled_sd > 0 else np.nan
else:
    cohen_d = np.nan

# Any affair (binary) chi-square
any_affair = (affairs > 0).astype(int)
cont_table = pd.crosstab(children, any_affair)
chi2_stat = chi2_p = np.nan
if cont_table.shape == (2, 2):
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(cont_table)

# Odds ratio for any affair (children yes vs no)
# Build 2x2: rows children yes/no, cols any_affair 1/0
odds_ratio = np.nan
try:
    # table values
    yes_affair = cont_table.loc["yes", 1]
    yes_noaff = cont_table.loc["yes", 0]
    no_affair = cont_table.loc["no", 1]
    no_noaff = cont_table.loc["no", 0]
    odds_ratio = (yes_affair / yes_noaff) / (no_affair / no_noaff)
except Exception:
    odds_ratio = np.nan

# Simple OLS regression (affairs ~ children yes/no)
# Encode children yes=1 no=0
model_results = None
try:
    df_reg = df.copy()
    df_reg["children_yes"] = (children == "yes").astype(int)
    # OLS
    ols = sm.OLS(df_reg["feature2"], sm.add_constant(df_reg["children_yes"])).fit()
    model_results = {
        "coef_children_yes": float(ols.params["children_yes"]),
        "p_value_children_yes": float(ols.pvalues["children_yes"]),
        "coef_intercept": float(ols.params["const"]),
        "r_squared": float(ols.rsquared),
    }
except Exception:
    model_results = None

# Logistic regression for any affair
logit_results = None
try:
    df_logit = df.copy()
    df_logit["children_yes"] = (children == "yes").astype(int)
    df_logit["any_affair"] = (df_logit["feature2"] > 0).astype(int)
    logit = sm.Logit(df_logit["any_affair"], sm.add_constant(df_logit["children_yes"])).fit(disp=False)
    logit_results = {
        "coef_children_yes": float(logit.params["children_yes"]),
        "p_value_children_yes": float(logit.pvalues["children_yes"]),
        "odds_ratio_children_yes": float(np.exp(logit.params["children_yes"])),
    }
except Exception:
    logit_results = None

results = {
    "summary": summary,
    "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
    "mannwhitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
    "cohen_d": float(cohen_d),
    "chi2_any_affair": {"chi2_stat": float(chi2_stat), "p_value": float(chi2_p)},
    "odds_ratio_any_affair_children_yes_vs_no": float(odds_ratio),
    "ols_affairs_children": model_results,
    "logit_any_affair_children": logit_results,
}

print(json.dumps(results, indent=2))
