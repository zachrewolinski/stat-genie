import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Filter to individuals with dyslexia (binary flag)
df = df[df["dyslexia_bin"] == 1].copy()

# Basic cleaning
for col in ["speed", "reader_view"]:
    df = df[pd.notnull(df[col])]

# Ensure positive speed
_df_before = len(df)
df = df[df["speed"] > 0]

# Create log speed
log_speed = np.log(df["speed"])
df["log_speed"] = log_speed

# Group stats
summary = {}
summary["n_total"] = int(len(df))
summary["n_reader_view_0"] = int((df["reader_view"] == 0).sum())
summary["n_reader_view_1"] = int((df["reader_view"] == 1).sum())

for rv in [0, 1]:
    g = df[df["reader_view"] == rv]["speed"]
    summary[f"mean_speed_rv_{rv}"] = float(g.mean())
    summary[f"median_speed_rv_{rv}"] = float(g.median())
    summary[f"std_speed_rv_{rv}"] = float(g.std(ddof=1))
    g_log = df[df["reader_view"] == rv]["log_speed"]
    summary[f"mean_log_speed_rv_{rv}"] = float(g_log.mean())

# Welch t-test on log speed
rv1 = df[df["reader_view"] == 1]["log_speed"]
rv0 = df[df["reader_view"] == 0]["log_speed"]

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, alternative="greater")
summary["welch_t_log_speed_stat"] = float(ttest.statistic)
summary["welch_t_log_speed_p_greater"] = float(ttest.pvalue)

# two-sided for reference
_ttest_two = stats.ttest_ind(rv1, rv0, equal_var=False, alternative="two-sided")
summary["welch_t_log_speed_p_two_sided"] = float(_ttest_two.pvalue)

# Mann-Whitney U on raw speed
try:
    mwu = stats.mannwhitneyu(
        df[df["reader_view"] == 1]["speed"],
        df[df["reader_view"] == 0]["speed"],
        alternative="greater",
    )
    summary["mwu_stat"] = float(mwu.statistic)
    summary["mwu_p_greater"] = float(mwu.pvalue)
except Exception as e:
    summary["mwu_error"] = str(e)

# Effect size (Cohen's d) on log speed
mean_diff = rv1.mean() - rv0.mean()
pooled_sd = np.sqrt(
    ((rv1.std(ddof=1) ** 2) + (rv0.std(ddof=1) ** 2)) / 2
)
summary["cohens_d_log"] = float(mean_diff / pooled_sd)
summary["mean_diff_log"] = float(mean_diff)
summary["log_speed_ratio"] = float(np.exp(mean_diff))  # multiplicative factor on speed

# Regression with controls (log speed)
# Use a parsimonious set to avoid overfitting; include page, device, age, gender, education, language, english_native
# Flesch_Kincaid and num_words capture text difficulty/length
formula = (
    "log_speed ~ reader_view + num_words + Flesch_Kincaid + age + "
    "C(page_id) + C(device) + C(gender) + C(education) + C(language) + C(english_native)"
)

model = smf.ols(formula, data=df).fit(cov_type="HC3")
summary["reg_coeff_reader_view"] = float(model.params.get("reader_view", np.nan))
summary["reg_se_reader_view"] = float(model.bse.get("reader_view", np.nan))
summary["reg_p_reader_view"] = float(model.pvalues.get("reader_view", np.nan))
summary["reg_ci_reader_view"] = [float(x) for x in model.conf_int().loc["reader_view"].tolist()]
summary["reg_n"] = int(model.nobs)
summary["reg_r2"] = float(model.rsquared)

# Save summary
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
