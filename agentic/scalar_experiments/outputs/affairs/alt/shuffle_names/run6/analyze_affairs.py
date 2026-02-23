import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
cwd = Path(__file__).resolve().parent

affairs_path = cwd / "affairs.csv"
df = pd.read_csv(affairs_path)

# According to info.json metadata in this task:
# - Column 'age' actually encodes extramarital intercourse frequency in the past year
# - Column 'religiousness' actually answers: "Are there children in the marriage?" (yes/no)
# So:
#   outcome: affair frequency coded as 0,1,2,3,7,12
#   main predictor: has_children (True/False) derived from 'religiousness'

# Sanity checks on expected columns
required_cols = {"age", "religiousness"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Construct variables
extramarital_freq = df["age"].astype(float)
# Binary: any affair in past year
any_affair = (extramarital_freq > 0).astype(int)

has_children = df["religiousness"].astype(str).str.lower().map({"yes": 1, "no": 0})
if has_children.isna().any():
    raise ValueError("Unexpected values in 'religiousness' when mapping to children indicator")

# Basic descriptive statistics
summary = {}

for label, mask in {"no_children": has_children == 0, "with_children": has_children == 1}.items():
    sub = extramarital_freq[mask]
    any_sub = any_affair[mask]
    summary[label] = {
        "n": int(mask.sum()),
        "mean_freq": float(sub.mean()),
        "median_freq": float(sub.median()),
        "prop_any_affair": float(any_sub.mean()),
    }

# Difference in mean affair frequency
mean_diff = summary["no_children"]["mean_freq"] - summary["with_children"]["mean_freq"]

# Two-sample t-test for difference in means
from scipy import stats

no_children_vals = extramarital_freq[has_children == 0]
with_children_vals = extramarital_freq[has_children == 1]

t_stat, t_pval = stats.ttest_ind(no_children_vals, with_children_vals, equal_var=False)

# Chi-squared test on binary any_affair vs has_children
contingency = pd.crosstab(has_children, any_affair)
chi2, chi2_pval, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression for any_affair ~ has_children + covariates
# Use covariates that are straightforwardly numeric in this version of the dataset.
X = pd.DataFrame({
    "intercept": 1.0,
    "has_children": has_children,
})

# Add additional numeric covariates where available
for col in ["education", "occupation", "children", "rating", "yearsmarried", "rownames"]:
    if col in df.columns:
        X[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing predictors
valid = X.notna().all(axis=1)
X_model = X.loc[valid]
y_model = any_affair.loc[valid]

logit_model = sm.Logit(y_model, X_model)
logit_result = logit_model.fit(disp=False)

coef_children = float(logit_result.params["has_children"])
se_children = float(logit_result.bse["has_children"])

# Wald z-test p-value for has_children coefficient
z_children = coef_children / se_children
p_children = float(2 * (1 - stats.norm.cdf(abs(z_children))))

# Compute marginal effect of having children on predicted probability of any affair
# at the mean of covariates.
mean_covariates = X_model.mean()
mean_covariates["has_children"] = 0

linpred_no_children = float(np.dot(mean_covariates.values, logit_result.params.values))
prob_no_children = float(1 / (1 + np.exp(-linpred_no_children)))

mean_covariates_with = mean_covariates.copy()
mean_covariates_with["has_children"] = 1
linpred_with_children = float(np.dot(mean_covariates_with.values, logit_result.params.values))
prob_with_children = float(1 / (1 + np.exp(-linpred_with_children)))

marginal_effect = prob_with_children - prob_no_children

# Collect key outputs for manual interpretation
results = {
    "summary": summary,
    "mean_diff_no_minus_with": mean_diff,
    "t_test_p_value": float(t_pval),
    "chi2_p_value": float(chi2_pval),
    "logit_coef_has_children": coef_children,
    "logit_p_value_has_children": p_children,
    "prob_any_affair_no_children_at_means": prob_no_children,
    "prob_any_affair_with_children_at_means": prob_with_children,
    "marginal_effect_children_on_prob": marginal_effect,
}

print(json.dumps(results, indent=2))
