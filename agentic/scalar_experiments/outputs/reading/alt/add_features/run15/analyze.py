import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Determine dyslexia indicator
if "dyslexia_bin" in df.columns and df["dyslexia_bin"].notna().any():
    dyslexia_flag = df["dyslexia_bin"] == 1
    dyslexia_source = "dyslexia_bin==1"
elif "dyslexia" in df.columns:
    dyslexia_flag = df["dyslexia"].fillna(0) >= 1
    dyslexia_source = "dyslexia>=1"
else:
    raise ValueError("No dyslexia indicator found")

# Filter dyslexia group and valid reader_view/speed
sub = df.loc[dyslexia_flag].copy()
sub = sub[sub["reader_view"].isin([0, 1])]
sub = sub[pd.to_numeric(sub["speed"], errors="coerce").notna()]
sub["speed"] = sub["speed"].astype(float)

# Remove non-positive speeds for log transform
sub = sub[sub["speed"] > 0]

n_rows = len(sub)

# Counts by condition
counts = sub["reader_view"].value_counts().to_dict()

# Participant counts
uuid_col = "uuid" if "uuid" in sub.columns else None
if uuid_col:
    participants_total = sub[uuid_col].nunique()
    participants_by_condition = sub.groupby("reader_view")[uuid_col].nunique().to_dict()
    both_conditions = sub.groupby(uuid_col)["reader_view"].nunique()
    participants_both = int((both_conditions == 2).sum())
else:
    participants_total = None
    participants_by_condition = None
    participants_both = None

# Descriptives
by_cond = sub.groupby("reader_view")["speed"]
mean_speed = by_cond.mean().to_dict()
median_speed = by_cond.median().to_dict()
std_speed = by_cond.std().to_dict()

# Welch t-test on log(speed)
log_speed = np.log(sub["speed"])
sub = sub.assign(log_speed=log_speed)

rv0 = sub[sub["reader_view"] == 0]["log_speed"]
rv1 = sub[sub["reader_view"] == 1]["log_speed"]

ttest_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney on raw speed
mw_res = stats.mannwhitneyu(
    sub[sub["reader_view"] == 1]["speed"],
    sub[sub["reader_view"] == 0]["speed"],
    alternative="two-sided",
)

# OLS with cluster-robust SE by uuid (if available)
ols_results = None
if uuid_col:
    # Include page_id as categorical control if present
    X = pd.DataFrame({"reader_view": sub["reader_view"]})
    if "page_id" in sub.columns:
        page_dummies = pd.get_dummies(sub["page_id"], prefix="page", drop_first=True)
        X = pd.concat([X, page_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(sub["log_speed"], X)
    ols_results = model.fit(cov_type="cluster", cov_kwds={"groups": sub[uuid_col]})

# Effect size on log scale (Cohen's d for log_speed)
mean_diff_log = rv1.mean() - rv0.mean()
pooled_sd = np.sqrt(((rv1.var(ddof=1) + rv0.var(ddof=1)) / 2))
cohens_d_log = mean_diff_log / pooled_sd if pooled_sd > 0 else np.nan

# Percent difference from log-scale mean diff
percent_diff = (np.exp(mean_diff_log) - 1.0) * 100.0

summary = {
    "dyslexia_source": dyslexia_source,
    "n_rows": int(n_rows),
    "counts": {str(k): int(v) for k, v in counts.items()},
    "participants_total": participants_total,
    "participants_by_condition": participants_by_condition,
    "participants_both_conditions": participants_both,
    "mean_speed": {str(k): float(v) for k, v in mean_speed.items()},
    "median_speed": {str(k): float(v) for k, v in median_speed.items()},
    "std_speed": {str(k): float(v) for k, v in std_speed.items()},
    "ttest_log": {
        "statistic": float(ttest_res.statistic),
        "pvalue": float(ttest_res.pvalue),
    },
    "mannwhitney": {
        "statistic": float(mw_res.statistic),
        "pvalue": float(mw_res.pvalue),
    },
    "log_mean_diff": float(mean_diff_log),
    "percent_diff": float(percent_diff),
    "cohens_d_log": float(cohens_d_log),
}

if ols_results is not None:
    coef = ols_results.params.get("reader_view", np.nan)
    pval = ols_results.pvalues.get("reader_view", np.nan)
    summary["ols_log_cluster"] = {
        "coef": float(coef),
        "pvalue": float(pval),
        "percent_diff": float((np.exp(coef) - 1.0) * 100.0),
        "n_obs": int(ols_results.nobs),
    }

# Paired analysis using participant-level means (if both conditions present)
if uuid_col and participants_both and participants_both > 0:
    per_participant = (
        sub.groupby([uuid_col, "reader_view"])["speed"]
        .mean()
        .unstack("reader_view")
        .dropna()
    )
    if 0 in per_participant.columns and 1 in per_participant.columns:
        # Paired t-test on log-speed means
        log0 = np.log(per_participant[0])
        log1 = np.log(per_participant[1])
        paired_t = stats.ttest_rel(log1, log0, nan_policy="omit")

        # Wilcoxon signed-rank on raw speed means
        try:
            wilcoxon_res = stats.wilcoxon(per_participant[1], per_participant[0])
            wilcoxon_stat = float(wilcoxon_res.statistic)
            wilcoxon_p = float(wilcoxon_res.pvalue)
        except ValueError:
            wilcoxon_stat = float("nan")
            wilcoxon_p = float("nan")

        mean_diff_log_paired = float((log1 - log0).mean())
        summary["paired_participant_means"] = {
            "n_participants": int(len(per_participant)),
            "log_mean_diff": mean_diff_log_paired,
            "percent_diff": float((np.exp(mean_diff_log_paired) - 1.0) * 100.0),
            "ttest_log": {
                "statistic": float(paired_t.statistic),
                "pvalue": float(paired_t.pvalue),
            },
            "wilcoxon_raw": {
                "statistic": wilcoxon_stat,
                "pvalue": wilcoxon_p,
            },
        }

print(json.dumps(summary, indent=2))
