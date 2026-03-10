import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("reading.csv")

# Define dyslexia group: dyslexia_bin == 1 (includes dyslexia and severe dyslexia)
_df = _df.copy()

# Basic cleaning
_df = _df.dropna(subset=["speed", "reader_view", "dyslexia_bin", "uuid", "page_id"])

# Filter dyslexia participants
_dys = _df[_df["dyslexia_bin"] == 1].copy()

# Ensure reader_view is binary
_dys = _dys[_dys["reader_view"].isin([0, 1])]

# Log-transform speed to reduce skew
_dys["log_speed"] = np.log(_dys["speed"].astype(float))

# Group stats
_groups = _dys.groupby("reader_view")
summary = _groups["speed"].agg(["count", "mean", "median", "std"]).rename(index={0: "no_reader_view", 1: "reader_view"})
summary_log = _groups["log_speed"].agg(["count", "mean", "median", "std"]).rename(index={0: "no_reader_view", 1: "reader_view"})

# Welch t-test on log speed
rv1 = _dys[_dys["reader_view"] == 1]["log_speed"].values
rv0 = _dys[_dys["reader_view"] == 0]["log_speed"].values

welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d) on log scale
n1, n0 = len(rv1), len(rv0)
var1, var0 = np.var(rv1, ddof=1), np.var(rv0, ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohen_d = (np.mean(rv1) - np.mean(rv0)) / pooled_sd if pooled_sd > 0 else np.nan

# Percent difference in geometric mean (exp of mean log speed)
geo_mean1 = np.exp(np.mean(rv1))
geo_mean0 = np.exp(np.mean(rv0))
geo_diff_pct = (geo_mean1 / geo_mean0 - 1) * 100.0

# Cluster-robust regression with page fixed effects
# log_speed ~ reader_view + C(page_id)
model = smf.ols("log_speed ~ reader_view + C(page_id)", data=_dys).fit(cov_type="cluster", cov_kwds={"groups": _dys["uuid"]})

# Within-subject paired analysis for users with both conditions
pivot = _dys.pivot_table(index="uuid", columns="reader_view", values="log_speed", aggfunc="mean")
paired = pivot.dropna()
paired_diff = paired[1] - paired[0]
if len(paired_diff) > 1:
    paired_test = stats.ttest_1samp(paired_diff, 0.0, nan_policy="omit")
else:
    paired_test = None

results = {
    "n_total_dyslexia_rows": int(len(_dys)),
    "n_unique_dyslexia_participants": int(_dys["uuid"].nunique()),
    "summary_speed": summary.to_dict(),
    "summary_log_speed": summary_log.to_dict(),
    "welch_ttest_log": {
        "t": float(welch.statistic),
        "p": float(welch.pvalue),
    },
    "cohen_d_log": float(cohen_d),
    "geo_mean_speed": {
        "reader_view": float(geo_mean1),
        "no_reader_view": float(geo_mean0),
        "percent_diff": float(geo_diff_pct),
    },
    "regression_reader_view_coef": {
        "coef": float(model.params.get("reader_view", np.nan)),
        "se": float(model.bse.get("reader_view", np.nan)),
        "p": float(model.pvalues.get("reader_view", np.nan)),
    },
    "paired_within_subject": {
        "n_pairs": int(len(paired_diff)),
        "mean_diff_log": float(paired_diff.mean()) if len(paired_diff) > 0 else np.nan,
        "t": float(paired_test.statistic) if paired_test is not None else np.nan,
        "p": float(paired_test.pvalue) if paired_test is not None else np.nan,
    },
}

print(json.dumps(results, indent=2))
