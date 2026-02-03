import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("affairs.csv")

# Clean: ensure children is categorical with yes/no
_df["children"] = _df["children"].astype(str).str.lower()

# Binary indicator for any affairs
_df["any_affair"] = (_df["affairs"] > 0).astype(int)

# Group stats
group_stats = _df.groupby("children")["affairs"].agg(["count", "mean", "median"])
any_stats = _df.groupby("children")["any_affair"].mean()

print("Affairs mean/median by children status:\n", group_stats)
print("\nProbability of any affair by children status:\n", any_stats)

# Difference in means (children yes - no)
mean_yes = group_stats.loc["yes", "mean"]
mean_no = group_stats.loc["no", "mean"]
mean_diff = mean_yes - mean_no

p_yes = any_stats.loc["yes"]
p_no = any_stats.loc["no"]
pp_diff = p_yes - p_no

print(f"\nMean difference (yes - no): {mean_diff:.3f}")
print(f"Probability difference (yes - no): {pp_diff:.3f}")

# Simple regression: OLS on affairs count with children indicator
_df["children_yes"] = (_df["children"] == "yes").astype(int)
ols = smf.ols("affairs ~ children_yes", data=_df).fit()
print("\nOLS affairs ~ children_yes")
print(ols.summary().tables[1])

# Logistic regression on any affair
logit = smf.logit("any_affair ~ children_yes", data=_df).fit(disp=False)
print("\nLogit any_affair ~ children_yes")
print(logit.summary().tables[1])

# Also control for basic covariates to check robustness
logit_ctrl = smf.logit(
    "any_affair ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating",
    data=_df,
).fit(disp=False)
print("\nLogit any_affair with controls")
print(logit_ctrl.summary().tables[1])

# Save key results to a small dict for convenience
results = {
    "mean_affairs_children_yes": float(mean_yes),
    "mean_affairs_children_no": float(mean_no),
    "mean_diff_yes_minus_no": float(mean_diff),
    "p_any_yes": float(p_yes),
    "p_any_no": float(p_no),
    "pp_diff_yes_minus_no": float(pp_diff),
    "ols_children_coef": float(ols.params["children_yes"]),
    "ols_children_pvalue": float(ols.pvalues["children_yes"]),
    "logit_children_coef": float(logit.params["children_yes"]),
    "logit_children_pvalue": float(logit.pvalues["children_yes"]),
    "logit_ctrl_children_coef": float(logit_ctrl.params["children_yes"]),
    "logit_ctrl_children_pvalue": float(logit_ctrl.pvalues["children_yes"]),
}

pd.Series(results).to_json("analysis_results.json", indent=2)
print("\nSaved results to analysis_results.json")
