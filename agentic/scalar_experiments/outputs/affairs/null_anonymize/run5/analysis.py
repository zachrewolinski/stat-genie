import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv("affairs.csv")

# Basic cleaning
# Ensure feature6 is binary 1 yes 0 no
child_map = {"yes": 1, "no": 0}

df["children"] = df["feature6"].map(child_map)

# Drop rows with missing in key columns
key_cols = ["feature2", "children", "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]

df_clean = df.dropna(subset=key_cols).copy()

# Summary stats
mean_yes = df_clean.loc[df_clean["children"] == 1, "feature2"].mean()
mean_no = df_clean.loc[df_clean["children"] == 0, "feature2"].mean()

# Two-sample t-test (Welch)

ttest = stats.ttest_ind(
    df_clean.loc[df_clean["children"] == 1, "feature2"],
    df_clean.loc[df_clean["children"] == 0, "feature2"],
    equal_var=False,
    nan_policy="omit",
)

# Mann-Whitney U (nonparam)
try:
    mwu = stats.mannwhitneyu(
        df_clean.loc[df_clean["children"] == 1, "feature2"],
        df_clean.loc[df_clean["children"] == 0, "feature2"],
        alternative="two-sided",
    )
except Exception:
    mwu = None

# OLS with controls
# feature3 is gender; treat as categorical
model_ols = smf.ols(
    "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df_clean,
).fit(cov_type="HC3")

# Poisson regression (count-like outcome) with robust SE
model_pois = smf.glm(
    "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df_clean,
    family=sm.families.Poisson(),
).fit(cov_type="HC3")

# Collect results
results = {
    "n": len(df_clean),
    "mean_children_yes": float(mean_yes),
    "mean_children_no": float(mean_no),
    "mean_diff_no_minus_yes": float(mean_no - mean_yes),
    "ttest_stat": float(ttest.statistic),
    "ttest_pvalue": float(ttest.pvalue),
    "mwu_stat": float(mwu.statistic) if mwu is not None else None,
    "mwu_pvalue": float(mwu.pvalue) if mwu is not None else None,
    "ols_coef_children": float(model_ols.params.get("children", np.nan)),
    "ols_pvalue_children": float(model_ols.pvalues.get("children", np.nan)),
    "pois_coef_children": float(model_pois.params.get("children", np.nan)),
    "pois_pvalue_children": float(model_pois.pvalues.get("children", np.nan)),
}

# Save for inspection
pd.Series(results).to_json("analysis_results.json", indent=2)

print(results)
