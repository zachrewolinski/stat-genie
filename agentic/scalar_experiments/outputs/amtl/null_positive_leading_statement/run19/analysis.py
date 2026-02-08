import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path("amtl.csv")

# Load data
_df = pd.read_csv(DATA_PATH)

# Basic cleaning/validation
_df = _df.copy()
_df = _df[_df["sockets"] > 0]
_df = _df[_df["num_amtl"].between(0, _df["sockets"]).fillna(False)]

# Human indicator
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Ensure categorical tooth_class
_df["tooth_class"] = _df["tooth_class"].astype("category")

# GLM binomial with counts
# Endog as two-column: successes, failures
_df["failures"] = _df["sockets"] - _df["num_amtl"]

formula = "num_amtl + failures ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial())
res = model.fit()

coef = res.params.get("is_human", np.nan)
se = res.bse.get("is_human", np.nan)

# z-score and p-value (two-sided)
if np.isfinite(coef) and np.isfinite(se) and se > 0:
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p = np.nan

# Effect size: odds ratio
odds_ratio = np.exp(coef) if np.isfinite(coef) else np.nan

# Map evidence to scalar in [-100, 100]
# Strength combines statistical signal (z) and magnitude (coef)
if np.isfinite(z) and np.isfinite(coef):
    z_strength = min(1.0, abs(z) / 4.0)  # z>=4 -> max
    mag_strength = min(1.0, abs(coef) / 1.0)  # log-odds >=1 -> max
    strength = 0.6 * z_strength + 0.4 * mag_strength
    score = int(round(100 * strength))
    if coef < 0:
        score = -score
else:
    score = 0

# Clamp just in case
score = max(-100, min(100, score))

# Save conclusion
Path("conclusion.txt").write_text(str(int(score)))

# Optional: write a small JSON report for transparency
report = {
    "n_rows": int(_df.shape[0]),
    "coef_is_human": float(coef) if np.isfinite(coef) else None,
    "se_is_human": float(se) if np.isfinite(se) else None,
    "z_is_human": float(z) if np.isfinite(z) else None,
    "p_is_human": float(p) if np.isfinite(p) else None,
    "odds_ratio_is_human": float(odds_ratio) if np.isfinite(odds_ratio) else None,
    "score": int(score),
}
Path("analysis_report.json").write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
