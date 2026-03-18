import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
csv_path = "affairs.csv"

df = pd.read_csv(csv_path)

# Variables
# feature2: frequency of affairs
# feature6: children yes/no

# Clean/prepare
# Ensure numeric for feature2
freq = pd.to_numeric(df["feature2"], errors="coerce")
children = df["feature6"].astype(str).str.lower().str.strip()

# Drop missing
data = df.copy()
data["freq"] = freq

data["children"] = children

# Only keep rows with valid children values and freq
valid = data["children"].isin(["yes", "no"]) & data["freq"].notna()

data = data.loc[valid].copy()

# Group stats
stats_by_children = (
    data.groupby("children")["freq"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

# T-test and Mann-Whitney U
freq_yes = data.loc[data["children"] == "yes", "freq"]
freq_no = data.loc[data["children"] == "no", "freq"]

ttest = stats.ttest_ind(freq_yes, freq_no, equal_var=False, nan_policy="omit")

# Mann-Whitney (nonparametric)
try:
    mwu = stats.mannwhitneyu(freq_yes, freq_no, alternative="two-sided")
except ValueError:
    mwu = None

# Effect size (Cohen's d)
# d = (mean1-mean2)/pooled std
mean_yes = freq_yes.mean()
mean_no = freq_no.mean()
std_yes = freq_yes.std(ddof=1)
std_no = freq_no.std(ddof=1)

n_yes = len(freq_yes)
n_no = len(freq_no)

pooled_std = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan

# Binary any-affair outcome
any_affair = (data["freq"] > 0).astype(int)

# Simple proportion test
prop_yes = any_affair[data["children"] == "yes"].mean()
prop_no = any_affair[data["children"] == "no"].mean()

# Two-proportion z-test
# statsmodels proportion z-test
count = np.array([
    any_affair[data["children"] == "yes"].sum(),
    any_affair[data["children"] == "no"].sum(),
])
obs = np.array([
    (data["children"] == "yes").sum(),
    (data["children"] == "no").sum(),
])

zstat, pval_prop = sm.stats.proportions_ztest(count, obs, alternative="two-sided")

# Logistic regression: any_affair ~ children + controls
# Controls: feature3 gender, feature4 age, feature5 yrs married, feature7 religiousness,
# feature8 education, feature9 occupation, feature10 marriage rating

# Prepare data for logit
model_df = data.copy()

# Encode children: yes=1, no=0
model_df["children_yes"] = (model_df["children"] == "yes").astype(int)

# Encode gender as binary: female=1, male=0 (if present)
model_df["female"] = (model_df["feature3"].astype(str).str.lower().str.strip() == "female").astype(int)

# Numeric controls
num_cols = ["feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]
for c in num_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

model_df["any_affair"] = (model_df["freq"] > 0).astype(int)

# Drop missing in model columns
model_cols = ["children_yes", "female"] + num_cols + ["any_affair"]
model_df = model_df.dropna(subset=model_cols)

X = model_df[["children_yes", "female"] + num_cols]
X = sm.add_constant(X)

y = model_df["any_affair"]

logit_result = sm.Logit(y, X).fit(disp=False)

# Extract children coefficient, OR, p-value
coef_children = logit_result.params["children_yes"]
se_children = logit_result.bse["children_yes"]
p_children = logit_result.pvalues["children_yes"]

odds_ratio = float(np.exp(coef_children))

# Linear regression: freq ~ children + controls (OLS)
X_lin = model_df[["children_yes", "female"] + num_cols]
X_lin = sm.add_constant(X_lin)

ols_result = sm.OLS(model_df["freq"], X_lin).fit()
coef_children_ols = ols_result.params["children_yes"]
p_children_ols = ols_result.pvalues["children_yes"]

# Save summary results to json for easy reading
results = {
    "n": int(len(data)),
    "group_stats": stats_by_children.to_dict(orient="records"),
    "ttest": {"stat": float(ttest.statistic), "p": float(ttest.pvalue)},
    "mannwhitney": None if mwu is None else {"stat": float(mwu.statistic), "p": float(mwu.pvalue)},
    "cohen_d": float(cohen_d),
    "prop_any_affair": {"children_yes": float(prop_yes), "children_no": float(prop_no)},
    "prop_test": {"z": float(zstat), "p": float(pval_prop)},
    "logit_children": {"coef": float(coef_children), "se": float(se_children), "p": float(p_children), "odds_ratio": float(odds_ratio)},
    "ols_children": {"coef": float(coef_children_ols), "p": float(p_children_ols)},
    "n_model": int(len(model_df))
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
