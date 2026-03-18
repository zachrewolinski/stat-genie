import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Identify columns
children_col = "feature6"  # yes/no
affairs_col = "feature2"   # numeric frequency of affairs

# Clean
sub = df[[children_col, affairs_col]].dropna()
sub[children_col] = sub[children_col].astype(str).str.lower().str.strip()

# Ensure only yes/no
sub = sub[sub[children_col].isin(["yes", "no"])]

# Define groups
with_children = sub[sub[children_col] == "yes"][affairs_col]
without_children = sub[sub[children_col] == "no"][affairs_col]

# Binary indicator: any affair
sub = sub.copy()
sub["any_affair"] = (sub[affairs_col] > 0).astype(int)
with_children_any = sub[sub[children_col] == "yes"]["any_affair"]
without_children_any = sub[sub[children_col] == "no"]["any_affair"]

# Descriptive stats
stats_desc = {
    "n_with_children": int(with_children.shape[0]),
    "n_without_children": int(without_children.shape[0]),
    "mean_with_children": float(with_children.mean()),
    "mean_without_children": float(without_children.mean()),
    "median_with_children": float(with_children.median()),
    "median_without_children": float(without_children.median()),
    "prop_any_with_children": float(with_children_any.mean()),
    "prop_any_without_children": float(without_children_any.mean()),
}

# Welch t-test on numeric frequency
welch_t = stats.ttest_ind(with_children, without_children, equal_var=False)

# Mann-Whitney U (non-parametric)
try:
    mw_u = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
except ValueError:
    mw_u = None

# Chi-square test for any affair
cont_table = pd.crosstab(sub[children_col], sub["any_affair"])
chi2, chi_p, chi_dof, chi_exp = stats.chi2_contingency(cont_table)

# Effect sizes
# Cohen's d (Welch-style: use pooled SD with unequal n)
mean_diff = with_children.mean() - without_children.mean()
var1 = with_children.var(ddof=1)
var2 = without_children.var(ddof=1)
pooled_sd = np.sqrt(((with_children.shape[0]-1)*var1 + (without_children.shape[0]-1)*var2) / (with_children.shape[0]+without_children.shape[0]-2))
cohens_d = float(mean_diff / pooled_sd) if pooled_sd > 0 else float("nan")

# Risk difference and odds ratio for any affair
p1 = with_children_any.mean()
p0 = without_children_any.mean()
risk_diff = float(p1 - p0)

# Odds ratio with Haldane-Anscombe correction if needed
ct = cont_table.copy()
if (ct == 0).any().any():
    ct = ct + 0.5
odds_ratio = float((ct.loc["yes",1] / ct.loc["yes",0]) / (ct.loc["no",1] / ct.loc["no",0]))

# Logistic regression (any affair ~ children)
# yes=1, no=0
sub["children_yes"] = (sub[children_col] == "yes").astype(int)
X = sm.add_constant(sub["children_yes"])
logit_model = sm.Logit(sub["any_affair"], X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues

results = {
    "descriptives": stats_desc,
    "welch_t": {"stat": float(welch_t.statistic), "p": float(welch_t.pvalue)},
    "mannwhitney": None if mw_u is None else {"stat": float(mw_u.statistic), "p": float(mw_u.pvalue)},
    "chi2_any_affair": {"chi2": float(chi2), "p": float(chi_p), "dof": int(chi_dof)},
    "effect_sizes": {
        "mean_diff": float(mean_diff),
        "cohens_d": float(cohens_d),
        "risk_diff": float(risk_diff),
        "odds_ratio": float(odds_ratio),
    },
    "logit_any_affair": {
        "coef_children_yes": float(logit_params["children_yes"]),
        "p_children_yes": float(logit_pvalues["children_yes"]),
    },
}

print(json.dumps(results, indent=2))

# Decide response based on evidence
# Heuristic: if children associated with lower affair frequency (negative mean_diff)
# and statistical tests indicate significance (p < 0.05) and effect size is meaningful.

mean_diff = results["effect_sizes"]["mean_diff"]
welch_p = results["welch_t"]["p"]
chi_p = results["chi2_any_affair"]["p"]
logit_p = results["logit_any_affair"]["p_children_yes"]

# Determine strength
# Base on consistency: if at least two tests significant and direction shows lower with children -> yes.

direction_lower = mean_diff < 0 and results["descriptives"]["prop_any_with_children"] < results["descriptives"]["prop_any_without_children"]

sig_tests = sum([welch_p < 0.05, chi_p < 0.05, logit_p < 0.05])

if direction_lower and sig_tests >= 2:
    # effect magnitude based on cohens d and risk diff
    d = abs(results["effect_sizes"]["cohens_d"])
    rd = abs(results["effect_sizes"]["risk_diff"])
    # map to Likert 60-90
    if d < 0.1 and rd < 0.02:
        response = 60
    elif d < 0.2 and rd < 0.05:
        response = 70
    elif d < 0.35 and rd < 0.10:
        response = 80
    else:
        response = 90
    answer = "Yes"
else:
    # No evidence or mixed
    if sig_tests == 0:
        response = 30
    elif sig_tests == 1:
        response = 40
    else:
        response = 50
    answer = "No"

explanation = {
    "answer": answer,
    "direction_lower_with_children": bool(direction_lower),
    "descriptives": stats_desc,
    "tests": {
        "welch_t_p": welch_p,
        "mannwhitney_p": None if results["mannwhitney"] is None else results["mannwhitney"]["p"],
        "chi2_any_affair_p": chi_p,
        "logit_any_affair_p": logit_p,
    },
    "effect_sizes": results["effect_sizes"],
}

conclusion = {
    "response": int(response),
    "explanation": json.dumps(explanation, ensure_ascii=True)
}

with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f, ensure_ascii=True)
