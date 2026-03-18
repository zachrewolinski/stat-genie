import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = "affairs.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure feature6 binary indicator: yes=1, no=0
children_map = {"yes": 1, "no": 0}
if df["feature6"].dtype == object:
    df["children"] = df["feature6"].map(children_map)
else:
    df["children"] = df["feature6"]

# Outcome
outcome = df["feature2"]

# Descriptive stats
summary = df.groupby("children")["feature2"].agg(["mean", "std", "count"])

# Two-sample t-test (Welch)
children_yes = df[df["children"] == 1]["feature2"].dropna()
children_no = df[df["children"] == 0]["feature2"].dropna()

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Effect size (Cohen's d, using pooled SD)
mean_yes = children_yes.mean()
mean_no = children_no.mean()
std_yes = children_yes.std(ddof=1)
std_no = children_no.std(ddof=1)

n_yes = len(children_yes)
n_no = len(children_no)
pooled_sd = np.sqrt(((n_yes - 1)*std_yes**2 + (n_no - 1)*std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression: outcome ~ children
ols_simple = smf.ols("feature2 ~ children", data=df).fit()

# OLS with controls (feature3-10 except outcome)
# Encode gender (feature3) as categorical
# Use categorical in formula
ols_controls = smf.ols(
    "feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit()

# Logistic regression for any affair (outcome > 0)
# If many negatives, >0 might still be meaningful; report as exploratory

df["affair_any"] = (df["feature2"] > 0).astype(int)
logit = smf.logit(
    "affair_any ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(disp=False)

# Compile results
result = {
    "summary_by_children": summary.to_dict(),
    "ttest_stat": float(ttest.statistic),
    "ttest_p": float(ttest.pvalue),
    "mean_yes": float(mean_yes),
    "mean_no": float(mean_no),
    "cohens_d": float(cohens_d),
    "ols_simple_coef": float(ols_simple.params.get("children", np.nan)),
    "ols_simple_p": float(ols_simple.pvalues.get("children", np.nan)),
    "ols_controls_coef": float(ols_controls.params.get("children", np.nan)),
    "ols_controls_p": float(ols_controls.pvalues.get("children", np.nan)),
    "logit_coef": float(logit.params.get("children", np.nan)),
    "logit_p": float(logit.pvalues.get("children", np.nan)),
    "logit_or": float(np.exp(logit.params.get("children", np.nan))),
}

print(result)
