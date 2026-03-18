import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv("affairs.csv")

# Ensure expected columns exist
expected_cols = [f"feature{i}" for i in range(1, 11)]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Variables
outcome = "feature2"  # frequency of extramarital affairs
children = "feature6"  # yes/no

df[children] = pd.Categorical(df[children], categories=["no", "yes"], ordered=False)

# Group stats
group_stats = (
    df.groupby(children)[outcome]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

# Split groups
y_no = df.loc[df[children] == "no", outcome].to_numpy()
y_yes = df.loc[df[children] == "yes", outcome].to_numpy()

# Welch t-test
welch_t = stats.ttest_ind(y_yes, y_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U
mw = stats.mannwhitneyu(y_yes, y_no, alternative="two-sided")

# Effect sizes
# Cohen's d (pooled)
mean_yes = np.nanmean(y_yes)
mean_no = np.nanmean(y_no)
std_yes = np.nanstd(y_yes, ddof=1)
std_no = np.nanstd(y_no, ddof=1)
pooled_sd = np.sqrt(((len(y_yes) - 1) * std_yes**2 + (len(y_no) - 1) * std_no**2) / (len(y_yes) + len(y_no) - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Cliff's delta
# Efficient computation using broadcasting can be heavy; use ranks formula
# delta = (2*U)/(n1*n2) - 1 with U being Mann-Whitney U for group1 vs group2
n1 = len(y_yes)
n2 = len(y_no)
# mw.statistic corresponds to U for y_yes
cliffs_delta = (2 * mw.statistic) / (n1 * n2) - 1

# Regression with controls
# Controls: gender, age, years married, religiousness, education, occupation, marriage rating
formula = (
    "feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + "
    "feature7 + feature8 + feature9 + feature10"
)
model = smf.ols(formula, data=df).fit(cov_type="HC3")

# Also log1p outcome
log_model = smf.ols(
    "np.log1p(feature2) ~ C(feature6) + C(feature3) + feature4 + feature5 + "
    "feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(cov_type="HC3")

# Extract coefficient for children yes vs no
coef_name = "C(feature6)[T.yes]"
coef = model.params.get(coef_name, np.nan)
pval = model.pvalues.get(coef_name, np.nan)
coef_log = log_model.params.get(coef_name, np.nan)
pval_log = log_model.pvalues.get(coef_name, np.nan)

results = {
    "group_stats": group_stats.to_dict(orient="records"),
    "welch_t_stat": welch_t.statistic,
    "welch_t_p": welch_t.pvalue,
    "mw_u": mw.statistic,
    "mw_p": mw.pvalue,
    "mean_yes": mean_yes,
    "mean_no": mean_no,
    "cohens_d": cohens_d,
    "cliffs_delta": cliffs_delta,
    "ols_coef_children_yes": coef,
    "ols_p_children_yes": pval,
    "ols_log_coef_children_yes": coef_log,
    "ols_log_p_children_yes": pval_log,
}

print(json.dumps(results, indent=2))
