import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv("amtl.csv")

# Ensure expected categories
_df["is_human"] = (_df["feature8"] == "Homo sapiens").astype(int)

# Basic counts
n_total = len(_df)

genus_counts = _df["feature8"].value_counts(dropna=False).to_dict()

# Unadjusted mean comparison
mean_human = _df.loc[_df["is_human"] == 1, "feature3"].mean()
mean_nonhuman = _df.loc[_df["is_human"] == 0, "feature3"].mean()
mean_diff = mean_human - mean_nonhuman

# OLS with robust SE, controlling for age, sex, tooth class
model = smf.ols(
    "feature3 ~ is_human + feature5 + feature7 + C(feature1)",
    data=_df,
).fit(cov_type="HC3")

coef = model.params.get("is_human", np.nan)
se = model.bse.get("is_human", np.nan)
pval = model.pvalues.get("is_human", np.nan)

# 95% CI
ci_low, ci_high = model.conf_int().loc["is_human"].tolist()

# Save results for inspection
results = {
    "n_total": int(n_total),
    "genus_counts": genus_counts,
    "mean_human": float(mean_human),
    "mean_nonhuman": float(mean_nonhuman),
    "mean_diff": float(mean_diff),
    "coef_is_human": float(coef),
    "se_is_human": float(se),
    "pval_is_human": float(pval),
    "ci_low": float(ci_low),
    "ci_high": float(ci_high),
    "r2": float(model.rsquared),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
