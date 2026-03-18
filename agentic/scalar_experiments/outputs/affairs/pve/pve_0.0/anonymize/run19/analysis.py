import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "affairs.csv"
df = pd.read_csv(csv_path)

# Map columns
# feature2: affairs frequency (numeric), feature6: children yes/no

# Basic cleaning
df = df.copy()

# Ensure expected columns
required_cols = ["feature2", "feature6", "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Create children indicator
# feature6 is categorical 'yes'/'no'
df["children"] = df["feature6"].astype(str).str.lower().map({"yes": 1, "no": 0})

# Drop rows with missing essentials
analysis_df = df.dropna(subset=["feature2", "children"]).copy()

# Descriptive stats by children
summary = analysis_df.groupby("children")["feature2"].agg([
    "count", "mean", "median", "std",
])

# Proportion with any affairs (>0)
analysis_df["any_affair"] = analysis_df["feature2"] > 0
prop_any = analysis_df.groupby("children")["any_affair"].mean()

# Two-sample t-test (Welch)
children_yes = analysis_df.loc[analysis_df["children"] == 1, "feature2"]
children_no = analysis_df.loc[analysis_df["children"] == 0, "feature2"]

welch_t = stats.ttest_ind(children_no, children_yes, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (nonparametric)
# Use two-sided test
mw = stats.mannwhitneyu(children_no, children_yes, alternative="two-sided")

# Effect size: difference in means and Cohen's d (pooled SD)
mean_diff = children_no.mean() - children_yes.mean()

# Cohen's d using pooled SD (unequal n)
pooled_sd = np.sqrt(
    ((children_no.var(ddof=1) * (len(children_no) - 1)) + (children_yes.var(ddof=1) * (len(children_yes) - 1)))
    / (len(children_no) + len(children_yes) - 2)
)
cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Regression: OLS on raw feature2 with controls
# Encode gender as binary (female/male); if other labels, use C()
ols_formula = (
    "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
ols_model = smf.ols(ols_formula, data=analysis_df).fit()
ols_model_robust = ols_model.get_robustcov_results(cov_type="HC3")

# Also logistic regression for any affair (cast to int)
analysis_df["any_affair_int"] = analysis_df["any_affair"].astype(int)
logit_formula = (
    "any_affair_int ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
logit_model = smf.logit(logit_formula, data=analysis_df).fit(disp=False)

results = {
    "summary": summary.reset_index().to_dict(orient="records"),
    "prop_any_affair": prop_any.reset_index().to_dict(orient="records"),
    "welch_t_stat": float(welch_t.statistic),
    "welch_t_pvalue": float(welch_t.pvalue),
    "mw_u_stat": float(mw.statistic),
    "mw_pvalue": float(mw.pvalue),
    "mean_diff_no_minus_yes": float(mean_diff),
    "cohens_d": float(cohens_d),
    "ols_children_coef": float(ols_model.params["children"]),
    "ols_children_pvalue": float(ols_model.pvalues["children"]),
    "ols_children_pvalue_hc3": float(ols_model_robust.pvalues[ols_model_robust.model.exog_names.index("children")]),
    "ols_r2": float(ols_model.rsquared),
    "logit_children_coef": float(logit_model.params["children"]),
    "logit_children_pvalue": float(logit_model.pvalues["children"]),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
