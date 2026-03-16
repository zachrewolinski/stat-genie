import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

pd.set_option("display.width", 200)

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Construct dyslexia flag
if "dyslexia" in df.columns:
    dyslexic = df["dyslexia"] >= 1
else:
    dyslexic = pd.Series(False, index=df.index)

if "dyslexia_bin" in df.columns:
    dyslexic = dyslexic | (df["dyslexia_bin"] == 1)

# Filter
dfd = df[dyslexic].copy()

# Keep relevant rows
dfd = dfd[dfd["reader_view"].notna() & dfd["speed"].notna()]

# Remove non-positive speeds for log transform
dfd = dfd[dfd["speed"] > 0]

# Basic counts
summary = {
    "n_total": len(df),
    "n_dyslexic_rows": len(dfd),
    "n_dyslexic_participants": dfd["uuid"].nunique(),
}

# Group stats
stats_rows = []
for rv, sub in dfd.groupby("reader_view"):
    stats_rows.append({
        "reader_view": int(rv),
        "n": len(sub),
        "participants": sub["uuid"].nunique(),
        "mean_speed": sub["speed"].mean(),
        "median_speed": sub["speed"].median(),
        "std_speed": sub["speed"].std(ddof=1),
    })

# Tests (raw speed)
rv1 = dfd[dfd["reader_view"] == 1]["speed"].dropna().to_numpy()
rv0 = dfd[dfd["reader_view"] == 0]["speed"].dropna().to_numpy()

test_results = {}
if len(rv1) >= 2 and len(rv0) >= 2:
    tt = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    test_results["welch_t_raw"] = {
        "t": float(tt.statistic),
        "p": float(tt.pvalue),
        "n1": len(rv1),
        "n0": len(rv0),
        "mean1": float(np.mean(rv1)),
        "mean0": float(np.mean(rv0)),
    }
    # Cohen's d
    s1 = np.var(rv1, ddof=1)
    s0 = np.var(rv0, ddof=1)
    pooled = np.sqrt(((len(rv1) - 1) * s1 + (len(rv0) - 1) * s0) / (len(rv1) + len(rv0) - 2))
    if pooled > 0:
        d = (np.mean(rv1) - np.mean(rv0)) / pooled
    else:
        d = np.nan
    test_results["cohens_d_raw"] = float(d)

# Log-speed tests (more robust to skew)
if len(rv1) >= 2 and len(rv0) >= 2:
    log1 = np.log(rv1)
    log0 = np.log(rv0)
    tlog = stats.ttest_ind(log1, log0, equal_var=False, nan_policy="omit")
    test_results["welch_t_log"] = {
        "t": float(tlog.statistic),
        "p": float(tlog.pvalue),
        "mean1": float(np.mean(log1)),
        "mean0": float(np.mean(log0)),
    }

# Nonparametric test
if len(rv1) >= 2 and len(rv0) >= 2:
    try:
        mw = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
        test_results["mannwhitney_raw"] = {
            "u": float(mw.statistic),
            "p": float(mw.pvalue),
        }
    except Exception as e:
        test_results["mannwhitney_raw"] = {"error": str(e)}

# Paired analysis for participants with both conditions
paired_results = {}
try:
    pivot = dfd.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
    paired = pivot.dropna()
    if paired.shape[0] >= 2:
        tpaired = stats.ttest_rel(paired[1], paired[0])
        paired_results["paired_t_raw"] = {
            "t": float(tpaired.statistic),
            "p": float(tpaired.pvalue),
            "n": int(paired.shape[0]),
            "mean_diff": float((paired[1] - paired[0]).mean()),
        }
    # log paired
    pivot_log = dfd.assign(log_speed=np.log(dfd["speed"]))
    pivot_log = pivot_log.pivot_table(index="uuid", columns="reader_view", values="log_speed", aggfunc="mean")
    paired_log = pivot_log.dropna()
    if paired_log.shape[0] >= 2:
        tpaired_log = stats.ttest_rel(paired_log[1], paired_log[0])
        paired_results["paired_t_log"] = {
            "t": float(tpaired_log.statistic),
            "p": float(tpaired_log.pvalue),
            "n": int(paired_log.shape[0]),
            "mean_diff": float((paired_log[1] - paired_log[0]).mean()),
        }
except Exception as e:
    paired_results["error"] = str(e)

# Mixed effects model on log speed
mixed_results = {}
try:
    dfd = dfd.copy()
    dfd["log_speed"] = np.log(dfd["speed"])
    if dfd["uuid"].nunique() >= 5:
        model = smf.mixedlm("log_speed ~ reader_view", dfd, groups=dfd["uuid"])
        fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        mixed_results = {
            "coef_reader_view": float(fit.params.get("reader_view", np.nan)),
            "p_reader_view": float(fit.pvalues.get("reader_view", np.nan)),
            "n_groups": int(dfd["uuid"].nunique()),
            "n_obs": int(dfd.shape[0]),
        }
except Exception as e:
    mixed_results["error"] = str(e)

print("SUMMARY", summary)
print("GROUP_STATS")
for row in stats_rows:
    print(row)
print("TESTS", test_results)
print("PAIRED", paired_results)
print("MIXED", mixed_results)
