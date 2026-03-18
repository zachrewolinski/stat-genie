import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic derived variables
# children: 1 if yes, 0 if no
if "feature6" not in df.columns:
    raise ValueError("Expected feature6 column not found")

df["children"] = (df["feature6"].astype(str).str.lower() == "yes").astype(int)

# Outcome: engagement in extramarital affairs (feature2 per metadata)
if "feature2" not in df.columns:
    raise ValueError("Expected feature2 column not found")

# Continuous outcome
outcome = df["feature2"].astype(float)

# Binary outcome: any positive engagement (threshold at > 0)
# Note: data is anonymized and may include negative values; this is a proxy indicator.
df["any_affair"] = (outcome > 0).astype(int)

# Descriptive stats by children
summary = df.groupby("children")["feature2"].agg(["count", "mean", "median", "std"]).reset_index()

# Welch t-test for difference in means
x = df.loc[df["children"] == 1, "feature2"].astype(float)
y = df.loc[df["children"] == 0, "feature2"].astype(float)

# Handle potential NaNs
x = x.dropna()
y = y.dropna()

t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)

# Mann-Whitney U test (nonparametric)
try:
    mw_u, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
except ValueError:
    # If ties or degenerate distributions cause issues, set to nan
    mw_u, mw_p = np.nan, np.nan

# Effect size: difference in means
mean_diff = x.mean() - y.mean()

# Cohen's d (unequal n, use pooled SD)
nx, ny = len(x), len(y)
var_x, var_y = x.var(ddof=1), y.var(ddof=1)
pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)) if (nx + ny - 2) > 0 else np.nan
cohens_d = mean_diff / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Chi-square test for any_affair vs children
contingency = pd.crosstab(df["children"], df["any_affair"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

# Odds ratio for any_affair (children vs no children)
# Add 0.5 continuity correction if any cell is zero
ct = contingency.values.astype(float)
if (ct == 0).any():
    ct = ct + 0.5
# table layout: rows children(0/1), cols any_affair(0/1)
# OR = (a/b) / (c/d) with a=child_yes & affair_yes, b=child_yes & affair_no, c=child_no & affair_yes, d=child_no & affair_no
# indices depend on crosstab ordering; ensure row/col labels
row_labels = list(contingency.index)
col_labels = list(contingency.columns)
# map to ensure correct positions
idx_child_yes = row_labels.index(1)
idx_child_no = row_labels.index(0)
idx_affair_yes = col_labels.index(1)
idx_affair_no = col_labels.index(0)

a = ct[idx_child_yes, idx_affair_yes]
b = ct[idx_child_yes, idx_affair_no]
c = ct[idx_child_no, idx_affair_yes]
d = ct[idx_child_no, idx_affair_no]

odds_ratio = (a / b) / (c / d)

# Logistic regression with controls (feature3 gender, feature4 age, feature5 years married, feature7 religiosity,
# feature8 education, feature9 occupation, feature10 marriage rating)
# Use only rows with non-missing values
# Encode categorical gender
if "feature3" in df.columns:
    df["gender"] = df["feature3"].astype("category")

formula = "any_affair ~ children"

# Add controls if available
controls = []
for col in ["feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]:
    if col in df.columns:
        if col == "feature3":
            controls.append("C(feature3)")
        else:
            controls.append(col)

if controls:
    formula = "any_affair ~ children + " + " + ".join(controls)

# Fit logistic regression; use try in case of separation issues
logit_res = None
try:
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    logit_res = {
        "coef_children": float(logit_model.params.get("children", np.nan)),
        "p_children": float(logit_model.pvalues.get("children", np.nan)),
        "odds_ratio_children": float(np.exp(logit_model.params.get("children", np.nan)))
    }
except Exception as e:
    logit_res = {"error": str(e)}

# OLS regression on continuous outcome with same controls
ols_res = None
try:
    ols_formula = "feature2 ~ children"
    if controls:
        ols_formula = "feature2 ~ children + " + " + ".join(controls)
    ols_model = smf.ols(formula=ols_formula, data=df).fit()
    ols_res = {
        "coef_children": float(ols_model.params.get("children", np.nan)),
        "p_children": float(ols_model.pvalues.get("children", np.nan))
    }
except Exception as e:
    ols_res = {"error": str(e)}

results = {
    "n": int(len(df)),
    "summary_by_children": summary.to_dict(orient="records"),
    "mean_diff_children_yes_minus_no": float(mean_diff),
    "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
    "t_test_p": float(t_p),
    "mannwhitney_p": float(mw_p) if not np.isnan(mw_p) else None,
    "any_affair_rates": {
        "children_yes": float(df.loc[df["children"] == 1, "any_affair"].mean()),
        "children_no": float(df.loc[df["children"] == 0, "any_affair"].mean())
    },
    "chi_square_p": float(chi_p),
    "odds_ratio_any_affair_children_vs_no": float(odds_ratio),
    "logit": logit_res,
    "ols": ols_res
}

print(json.dumps(results, indent=2))
