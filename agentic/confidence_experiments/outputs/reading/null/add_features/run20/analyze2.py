import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

if "dyslexia_bin" in df.columns:
    dys = df["dyslexia_bin"].copy()
else:
    dys = (df["dyslexia"] > 0).astype(int)

sub = df.loc[dys == 1].copy()

required_cols = ["speed", "reader_view", "uuid", "page_id"]
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sub = sub.dropna(subset=["speed", "reader_view", "uuid", "page_id"]).copy()
sub = sub[sub["speed"] > 0].copy()
sub["reader_view"] = sub["reader_view"].astype(int)
sub["page_id"] = sub["page_id"].astype("category")
sub["log_speed"] = np.log(sub["speed"])

# Descriptive stats

desc = sub.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Welch t-test on log speed
rv0 = sub.loc[sub["reader_view"] == 0, "log_speed"]
rv1 = sub.loc[sub["reader_view"] == 1, "log_speed"]

t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney U on raw speed
try:
    u_stat, u_p = stats.mannwhitneyu(
        sub.loc[sub["reader_view"] == 1, "speed"],
        sub.loc[sub["reader_view"] == 0, "speed"],
        alternative="two-sided"
    )
except ValueError:
    u_stat, u_p = np.nan, np.nan

# OLS with page fixed effects and cluster-robust SE by uuid
ols = smf.ols("log_speed ~ reader_view + C(page_id)", data=sub).fit(
    cov_type="cluster", cov_kwds={"groups": sub["uuid"]}
)
coef = ols.params.get("reader_view", np.nan)
pval = ols.pvalues.get("reader_view", np.nan)

pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

result = {
    "n_rows": int(len(sub)),
    "n_uuid": int(sub["uuid"].nunique()),
    "desc": desc.to_dict(orient="records"),
    "ttest_log": {"t": float(t_stat), "p": float(t_p)},
    "mannwhitney": {"u": float(u_stat) if np.isfinite(u_stat) else None, "p": float(u_p) if np.isfinite(u_p) else None},
    "ols_cluster": {
        "coef_log": float(coef) if np.isfinite(coef) else None,
        "p": float(pval) if np.isfinite(pval) else None,
        "pct_change": float(pct_change) if np.isfinite(pct_change) else None,
    },
}

print(json.dumps(result, indent=2))
