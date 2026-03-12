import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Load data
csv_path = "reading.csv"
df = pd.read_csv(csv_path)

# Filter to participants with dyslexia (binary indicator)
df = df[df["dyslexia_bin"] == 1].copy()

# Keep positive speeds only
if "speed" in df.columns:
    df = df[df["speed"].notna() & (df["speed"] > 0)].copy()

if len(df) == 0:
    raise SystemExit("No data after filtering for dyslexia and positive speed.")

# Log-transform speed
df["log_speed"] = np.log(df["speed"])

# Summary statistics by reader_view
summary_speed = df.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()
summary_log = df.groupby("reader_view")["log_speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Welch t-test on log speed
rv1 = df[df["reader_view"] == 1]["log_speed"].values
rv0 = df[df["reader_view"] == 0]["log_speed"].values
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d for independent groups)
mean1, mean0 = np.nanmean(rv1), np.nanmean(rv0)
std1, std0 = np.nanstd(rv1, ddof=1), np.nanstd(rv0, ddof=1)
pooled = np.sqrt((std1**2 + std0**2) / 2)
cohens_d = (mean1 - mean0) / pooled if pooled > 0 else np.nan

# Paired analysis: within-subject means by reader_view
paired = (
    df.groupby(["uuid", "reader_view"])["log_speed"].mean().unstack()
)
paired = paired.dropna()
paired_n = int(len(paired))
paired_t = None
paired_d = None
paired_diff_mean = None
if paired_n >= 2:
    diff = paired[1] - paired[0]
    paired_diff_mean = diff.mean()
    paired_t = stats.ttest_rel(paired[1], paired[0])
    paired_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

# Mixed effects model (random intercept for participant)
mixed_ok = False
mixed_result = None
mixed_error = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    try:
        model = smf.mixedlm(
            "log_speed ~ reader_view + C(page_id) + C(device)",
            data=df,
            groups=df["uuid"],
        )
        mixed_result = model.fit(reml=False, method="lbfgs")
        mixed_ok = True
    except Exception as e:
        mixed_error = str(e)
        mixed_ok = False

# OLS with cluster-robust SEs by uuid (align groups to rows used in model)
ols_result = None
if not mixed_ok:
    ols_model = smf.ols(
        "log_speed ~ reader_view + C(page_id) + C(device)",
        data=df,
    )
    # Align groups to model's row labels (drops missing internally)
    groups = df.loc[ols_model.data.row_labels, "uuid"]
    ols_result = ols_model.fit(cov_type="cluster", cov_kwds={"groups": groups})

output = {
    "n_rows_dyslexia": int(len(df)),
    "n_participants_dyslexia": int(df["uuid"].nunique()),
    "summary_speed": summary_speed.to_dict(orient="records"),
    "summary_log_speed": summary_log.to_dict(orient="records"),
    "welch_t": {
        "statistic": float(welch_t.statistic),
        "pvalue": float(welch_t.pvalue),
        "cohens_d": float(cohens_d),
        "mean_log_speed_reader_view_1": float(mean1),
        "mean_log_speed_reader_view_0": float(mean0),
    },
    "paired": {
        "n_pairs": paired_n,
        "mean_log_diff": None if paired_diff_mean is None else float(paired_diff_mean),
        "ttest_rel_statistic": None if paired_t is None else float(paired_t.statistic),
        "ttest_rel_pvalue": None if paired_t is None else float(paired_t.pvalue),
        "cohens_d_paired": None if paired_d is None else float(paired_d),
    },
    "mixed_model": None,
    "ols_cluster": None,
}

if mixed_ok and mixed_result is not None:
    output["mixed_model"] = {
        "reader_view_coef": float(mixed_result.params.get("reader_view", np.nan)),
        "reader_view_pvalue": float(mixed_result.pvalues.get("reader_view", np.nan)),
        "reader_view_se": float(mixed_result.bse.get("reader_view", np.nan)),
        "model_converged": bool(getattr(mixed_result, "converged", False)),
        "n_obs": int(mixed_result.nobs),
    }
else:
    output["mixed_model"] = {"error": mixed_error}

if ols_result is not None:
    output["ols_cluster"] = {
        "reader_view_coef": float(ols_result.params.get("reader_view", np.nan)),
        "reader_view_pvalue": float(ols_result.pvalues.get("reader_view", np.nan)),
        "reader_view_se": float(ols_result.bse.get("reader_view", np.nan)),
        "n_obs": int(ols_result.nobs),
    }

print(json.dumps(output, indent=2))
