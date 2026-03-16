import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Ensure numeric speed and reader_view, dyslexia_bin
for col in ["speed", "reader_view", "dyslexia_bin", "dyslexia"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter dyslexia individuals
if "dyslexia_bin" in df.columns:
    dys = df[df["dyslexia_bin"] == 1].copy()
else:
    dys = df[df["dyslexia"] > 0].copy()

# Drop missing speed or reader_view
dys = dys.dropna(subset=["speed", "reader_view", "uuid", "page_id"])

# Descriptive stats by reader_view
summary = dys.groupby("reader_view").agg(
    n=("speed", "size"),
    mean_speed=("speed", "mean"),
    median_speed=("speed", "median"),
    std_speed=("speed", "std"),
).reset_index()

# Nonparametric test (Mann-Whitney U) for speed
rv0 = dys[dys["reader_view"] == 0]["speed"]
rv1 = dys[dys["reader_view"] == 1]["speed"]

mw_res = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

# Effect size for MW: rank-biserial
n1, n0 = len(rv1), len(rv0)
U = mw_res.statistic
# rank-biserial = 1 - 2U/(n1*n0)
rank_biserial = 1 - 2 * U / (n1 * n0)

# Log transform for modeling to reduce skew
# add small constant to handle zeros if any
log_speed = np.log(dys["speed"] + 1)

dys = dys.assign(log_speed=log_speed)

# Mixed effects model: log_speed ~ reader_view + C(page_id)
# random intercept by uuid
model = smf.mixedlm("log_speed ~ reader_view + C(page_id)", dys, groups=dys["uuid"])
try:
    result = model.fit(reml=False, method="lbfgs")
except Exception:
    result = model.fit(reml=False)

# Extract reader_view effect
coef = result.params.get("reader_view", np.nan)
se = result.bse.get("reader_view", np.nan)
# Wald z-test
z = coef / se if se != 0 else np.nan
p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Convert log effect to percent change
pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

out = {
    "n_dyslexia": int(dys.shape[0]),
    "n_reader_view_0": int(n0),
    "n_reader_view_1": int(n1),
    "summary": summary.to_dict(orient="records"),
    "mannwhitney_u": float(U),
    "mannwhitney_p": float(mw_res.pvalue),
    "rank_biserial": float(rank_biserial),
    "mixedlm_coef_log": float(coef),
    "mixedlm_se": float(se),
    "mixedlm_z": float(z),
    "mixedlm_p": float(p),
    "mixedlm_pct_change": float(pct_change),
}

print(json.dumps(out, indent=2))
