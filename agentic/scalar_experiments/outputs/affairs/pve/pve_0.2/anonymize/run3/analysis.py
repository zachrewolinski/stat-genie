import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Variables
# feature2: extramarital affairs frequency (continuous)
# feature6: children yes/no

df = df.copy()

# Clean children indicator
df["children"] = df["feature6"].map({"yes": 1, "no": 0})

# Basic checks
n_total = len(df)
children_counts = df["children"].value_counts(dropna=False).to_dict()

# Outcome
outcome = df["feature2"]

# Group stats
stats_by_children = df.groupby("children")["feature2"].agg(["count", "mean", "std", "median"]).reset_index()

# Welch t-test
no_vals = df.loc[df["children"] == 0, "feature2"].dropna()
yes_vals = df.loc[df["children"] == 1, "feature2"].dropna()

ttest = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy="omit")

# Cohen's d (using pooled SD with unequal sizes)
mean_yes = yes_vals.mean()
mean_no = no_vals.mean()
std_yes = yes_vals.std(ddof=1)
std_no = no_vals.std(ddof=1)

n_yes = yes_vals.shape[0]
n_no = no_vals.shape[0]

pooled_sd = np.sqrt(((n_yes - 1) * std_yes ** 2 + (n_no - 1) * std_no ** 2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd != 0 else np.nan

# Mann-Whitney U
mw = stats.mannwhitneyu(yes_vals, no_vals, alternative="two-sided")

# Regression with controls
# feature3 gender, feature4 age, feature5 years married, feature7 religiousness, feature8 education,
# feature9 occupation, feature10 marriage rating

# Encode gender as categorical

# OLS with robust SEs
formula = "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
model = smf.ols(formula, data=df).fit(cov_type="HC3")

children_coef = model.params.get("children", np.nan)
children_p = model.pvalues.get("children", np.nan)
children_ci = model.conf_int().loc["children"].tolist() if "children" in model.params else [np.nan, np.nan]

# Logistic regression: any affair
# Define any affair as feature2 > 0
# (If feature2 contains negative values from anonymization, this keeps 0 as none)

df["affair_any"] = (df["feature2"] > 0).astype(int)

logit_formula = "affair_any ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

logit_children_coef = logit_model.params.get("children", np.nan)
logit_children_p = logit_model.pvalues.get("children", np.nan)
logit_children_or = np.exp(logit_children_coef) if np.isfinite(logit_children_coef) else np.nan
logit_children_ci = logit_model.conf_int().loc["children"].tolist() if "children" in logit_model.params else [np.nan, np.nan]
logit_children_or_ci = [float(np.exp(logit_children_ci[0])), float(np.exp(logit_children_ci[1]))] if all(np.isfinite(logit_children_ci)) else [np.nan, np.nan]

results = {
    "n_total": int(n_total),
    "children_counts": {str(k): int(v) for k, v in children_counts.items()},
    "group_stats": stats_by_children.to_dict(orient="records"),
    "ttest": {
        "statistic": float(ttest.statistic),
        "pvalue": float(ttest.pvalue),
        "mean_yes": float(mean_yes),
        "mean_no": float(mean_no),
        "cohens_d": float(cohens_d),
    },
    "mannwhitney": {
        "statistic": float(mw.statistic),
        "pvalue": float(mw.pvalue),
    },
    "ols": {
        "children_coef": float(children_coef),
        "children_p": float(children_p),
        "children_ci": [float(children_ci[0]), float(children_ci[1])],
    },
    "logit": {
        "children_coef": float(logit_children_coef),
        "children_p": float(logit_children_p),
        "children_or": float(logit_children_or),
        "children_or_ci": [float(logit_children_or_ci[0]), float(logit_children_or_ci[1])],
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
