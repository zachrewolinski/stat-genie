import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Verify reading speed relation (feature20) ~ words per minute using feature5 (reading time minus scrolling)
# wpm = words / (time_minutes)
# time in ms -> minutes = ms / 60000
with np.errstate(divide='ignore', invalid='ignore'):
    df["calc_wpm"] = df["feature7"] / (df["feature5"] / 60000.0)

corr = df[["feature20", "calc_wpm"]].corr().iloc[0, 1]

# Primary dyslexia indicator
# feature17: has dyslexia (1) or not (0)
df_dys = df[df["feature17"] == 1].copy()

# Basic counts
n_obs = len(df_dys)
participants = df_dys["feature1"].nunique()

# Group stats by reader view (feature3)
summary = (
    df_dys.groupby("feature3")["feature20"]
    .agg(["count", "mean", "std"])\
    .rename(index={0: "NoReaderView", 1: "ReaderView"})
)

# Independent (cluster-robust) regression with participant clustering
ols = smf.ols("feature20 ~ feature3", data=df_dys).fit(
    cov_type="cluster", cov_kwds={"groups": df_dys["feature1"]}
)

# Mixed effects model (random intercept per participant)
try:
    mixed = smf.mixedlm("feature20 ~ feature3", df_dys, groups=df_dys["feature1"]).fit(reml=False)
except Exception as e:
    mixed = None

# Paired t-test on participant means where both conditions are present
pivot = df_dys.pivot_table(index="feature1", columns="feature3", values="feature20", aggfunc="mean")
paired = pivot.dropna()
paired_n = len(paired)
if paired_n > 1:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    paired_diff = (paired[1] - paired[0]).mean()
    paired_sd = (paired[1] - paired[0]).std(ddof=1)
else:
    t_stat, p_val, paired_diff, paired_sd = np.nan, np.nan, np.nan, np.nan

# Effect size (Cohen's d for independent groups)
mean1 = summary.loc["ReaderView", "mean"] if "ReaderView" in summary.index else np.nan
mean0 = summary.loc["NoReaderView", "mean"] if "NoReaderView" in summary.index else np.nan
std1 = summary.loc["ReaderView", "std"] if "ReaderView" in summary.index else np.nan
std0 = summary.loc["NoReaderView", "std"] if "NoReaderView" in summary.index else np.nan
n1 = summary.loc["ReaderView", "count"] if "ReaderView" in summary.index else np.nan
n0 = summary.loc["NoReaderView", "count"] if "NoReaderView" in summary.index else np.nan

if all(np.isfinite([std1, std0, n1, n0])) and n1 > 1 and n0 > 1:
    pooled = np.sqrt(((n1 - 1) * std1**2 + (n0 - 1) * std0**2) / (n1 + n0 - 2))
    cohend = (mean1 - mean0) / pooled if pooled > 0 else np.nan
else:
    cohend = np.nan

# Sensitivity: include severe dyslexia (feature12 > 0)
df_dys2 = df[df["feature12"] > 0].copy()
summary2 = df_dys2.groupby("feature3")["feature20"].agg(["count", "mean", "std"]).rename(index={0: "NoReaderView", 1: "ReaderView"})
ols2 = smf.ols("feature20 ~ feature3", data=df_dys2).fit(
    cov_type="cluster", cov_kwds={"groups": df_dys2["feature1"]}
)

results = {
    "corr_feature20_calc_wpm": corr,
    "dyslexia_feature17": {
        "n_obs": int(n_obs),
        "n_participants": int(participants),
        "summary": summary.reset_index().to_dict(orient="records"),
        "ols_coef": float(ols.params.get("feature3", np.nan)),
        "ols_pvalue": float(ols.pvalues.get("feature3", np.nan)),
        "ols_ci": [float(x) for x in ols.conf_int().loc["feature3"].tolist()],
        "mixed_coef": float(mixed.params.get("feature3", np.nan)) if mixed is not None else np.nan,
        "mixed_pvalue": float(mixed.pvalues.get("feature3", np.nan)) if mixed is not None else np.nan,
        "paired_n": int(paired_n),
        "paired_mean_diff": float(paired_diff),
        "paired_t": float(t_stat),
        "paired_p": float(p_val),
        "paired_sd": float(paired_sd),
        "cohen_d": float(cohend),
    },
    "dyslexia_feature12_gt0": {
        "n_obs": int(len(df_dys2)),
        "n_participants": int(df_dys2["feature1"].nunique()),
        "summary": summary2.reset_index().to_dict(orient="records"),
        "ols_coef": float(ols2.params.get("feature3", np.nan)),
        "ols_pvalue": float(ols2.pvalues.get("feature3", np.nan)),
        "ols_ci": [float(x) for x in ols2.conf_int().loc["feature3"].tolist()],
    },
}

print(json.dumps(results, indent=2))
