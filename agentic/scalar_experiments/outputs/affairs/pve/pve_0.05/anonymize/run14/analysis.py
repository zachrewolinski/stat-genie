import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Identify columns
# feature2: affairs engagement (numeric)
# feature6: children yes/no (category)

# Clean
outcome = "feature2"
children = "feature6"

# Ensure no missing
analysis_df = df[[outcome, children, "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].dropna()

# Encode children as binary: yes=1, no=0
analysis_df["children_yes"] = (analysis_df[children].astype(str).str.lower() == "yes").astype(int)

# Descriptive stats
by_children = analysis_df.groupby("children_yes")[outcome].agg(["count", "mean", "std", "median"])

# t-test for difference in means
vals_yes = analysis_df.loc[analysis_df["children_yes"] == 1, outcome]
vals_no = analysis_df.loc[analysis_df["children_yes"] == 0, outcome]

# Welch t-test
ttest = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d, using pooled SD)
mean_yes = vals_yes.mean()
mean_no = vals_no.mean()
std_yes = vals_yes.std(ddof=1)
std_no = vals_no.std(ddof=1)

n_yes = vals_yes.shape[0]
n_no = vals_no.shape[0]
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd != 0 else np.nan

# OLS regression with controls
X = analysis_df[["children_yes", "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].copy()
# Encode gender if categorical
X["feature3"] = X["feature3"].astype(str)
X = pd.get_dummies(X, columns=["feature3"], drop_first=True)

X = sm.add_constant(X)
model = sm.OLS(analysis_df[outcome], X).fit(cov_type="HC3")

result = {
    "n": int(analysis_df.shape[0]),
    "by_children": by_children.to_dict(),
    "t_test": {
        "statistic": float(ttest.statistic),
        "pvalue": float(ttest.pvalue),
        "mean_yes": float(mean_yes),
        "mean_no": float(mean_no),
        "cohen_d": float(cohen_d),
    },
    "regression_children_coef": float(model.params.get("children_yes", np.nan)),
    "regression_children_pvalue": float(model.pvalues.get("children_yes", np.nan)),
    "regression_children_ci": list(map(float, model.conf_int().loc["children_yes"].tolist())),
    "regression_summary": {
        "r2": float(model.rsquared),
        "n": int(model.nobs),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
