import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "affairs.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Basic info
n_rows, n_cols = _df.shape

# Identify columns
cols = list(_df.columns)

# Variables based on metadata
outcome = "feature2"  # engagement in extramarital affairs (numeric)
children_col = "feature6"  # yes/no

# Clean / encode
# Drop rows with missing outcome or children
analysis_df = _df.dropna(subset=[outcome, children_col]).copy()

# Map children to binary
child_map = {"yes": 1, "no": 0, "Yes": 1, "No": 0, "YES": 1, "NO": 0}
analysis_df["children"] = analysis_df[children_col].map(child_map)

# Some datasets may already be numeric or encoded
if analysis_df["children"].isna().any():
    # try if already 0/1
    if pd.api.types.is_numeric_dtype(analysis_df[children_col]):
        analysis_df["children"] = analysis_df[children_col]
    else:
        # try lowercasing
        analysis_df["children"] = analysis_df[children_col].astype(str).str.lower().map({"yes": 1, "no": 0})

analysis_df = analysis_df.dropna(subset=["children"])
analysis_df["children"] = analysis_df["children"].astype(int)

# Group summaries
summary = analysis_df.groupby("children")[outcome].agg(["count", "mean", "median", "std"]).reset_index()

# Welch t-test for mean difference
child_yes = analysis_df.loc[analysis_df["children"] == 1, outcome]
child_no = analysis_df.loc[analysis_df["children"] == 0, outcome]

t_stat, p_val = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy="omit")

# Effect size: Cohen's d (pooled SD)
# Use unbiased pooled SD (Welch-style weighting)
ny = child_yes.shape[0]
no = child_no.shape[0]
sy = child_yes.std(ddof=1)
so = child_no.std(ddof=1)
pooled_sd = np.sqrt(((ny - 1) * sy**2 + (no - 1) * so**2) / (ny + no - 2)) if (ny + no - 2) > 0 else np.nan
cohens_d = (child_yes.mean() - child_no.mean()) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# OLS with controls
# Controls: gender (feature3), age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)
# Use robust (HC3) SE
formula = (
    "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
reg_df = analysis_df.dropna(
    subset=[
        "children",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    ]
).copy()

ols_model = smf.ols(formula, data=reg_df).fit(cov_type="HC3")

children_coef = ols_model.params.get("children", np.nan)
children_p = ols_model.pvalues.get("children", np.nan)

# Logistic regression on any-affair indicator (feature2 > 0)
# If outcome has negative values due to anonymization, still use >0 threshold as proxy.
reg_df["any_affair"] = (reg_df["feature2"] > 0).astype(int)
logit_model = smf.logit(
    "any_affair ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=reg_df,
).fit(disp=False)

logit_coef = logit_model.params.get("children", np.nan)
logit_p = logit_model.pvalues.get("children", np.nan)
# Convert to odds ratio
logit_or = float(np.exp(logit_coef)) if np.isfinite(logit_coef) else np.nan

results = {
    "n_rows": n_rows,
    "n_cols": n_cols,
    "columns": cols,
    "summary_by_children": summary.to_dict(orient="records"),
    "ttest": {"t_stat": float(t_stat), "p_value": float(p_val)},
    "cohens_d": float(cohens_d),
    "ols": {"coef": float(children_coef), "p_value": float(children_p), "n": int(reg_df.shape[0])},
    "logit": {"coef": float(logit_coef), "p_value": float(logit_p), "odds_ratio": float(logit_or)},
}

print(json.dumps(results, indent=2))
