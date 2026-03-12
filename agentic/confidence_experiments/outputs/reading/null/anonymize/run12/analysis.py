import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Rename columns to more readable names based on info.json
col_map = {
    "feature1": "participant_id",
    "feature2": "page_id",
    "feature3": "reader_view",
    "feature4": "time_total_ms",
    "feature5": "time_reading_ms",
    "feature6": "time_scrolling_ms",
    "feature7": "word_count",
    "feature8": "comprehension_rate",
    "feature9": "image_width",
    "feature10": "age",
    "feature11": "device",
    "feature12": "dyslexia_status_3level",
    "feature13": "education",
    "feature14": "gender",
    "feature15": "language",
    "feature16": "retake",
    "feature17": "dyslexia_binary",
    "feature18": "native_english",
    "feature19": "flesch_kincaid",
    "feature20": "reading_speed",
}

# Ensure all columns exist
missing = [c for c in col_map if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.rename(columns=col_map)

# Check if reading_speed matches computed words per minute
# Compute two candidate speeds
speed_total = df["word_count"] * 60000.0 / df["time_total_ms"]
speed_reading = df["word_count"] * 60000.0 / df["time_reading_ms"]

# Correlations (drop inf and nan)
valid = np.isfinite(df["reading_speed"])
valid_total = valid & np.isfinite(speed_total)
valid_read = valid & np.isfinite(speed_reading)

corr_total = np.corrcoef(df.loc[valid_total, "reading_speed"], speed_total[valid_total])[0, 1]
corr_read = np.corrcoef(df.loc[valid_read, "reading_speed"], speed_reading[valid_read])[0, 1]

# We'll use the higher correlation to infer what reading_speed is based on
speed_source = "time_total_ms" if corr_total >= corr_read else "time_reading_ms"

# Subset to dyslexia individuals (binary dyslexia)
# Use feature17 (dyslexia_binary) == 1
sub = df[df["dyslexia_binary"] == 1].copy()

# Drop rows with missing reading_speed or reader_view
sub = sub[np.isfinite(sub["reading_speed"]) & np.isfinite(sub["reader_view"])]

# Ensure reader_view is binary 0/1
sub = sub[sub["reader_view"].isin([0, 1])]

# Basic group stats
rv0 = sub[sub["reader_view"] == 0]["reading_speed"]
rv1 = sub[sub["reader_view"] == 1]["reading_speed"]

n0 = rv0.shape[0]
n1 = rv1.shape[0]

mean0 = rv0.mean()
mean1 = rv1.mean()

# Welch's t-test
if n0 > 1 and n1 > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
else:
    t_stat, p_val = np.nan, np.nan

# Effect size (Cohen's d, using pooled SD with unequal n)
# Use Hedges g correction to be conservative
s0 = rv0.std(ddof=1)
s1 = rv1.std(ddof=1)
if n0 > 1 and n1 > 1 and np.isfinite(s0) and np.isfinite(s1):
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    d = (mean1 - mean0) / pooled if pooled > 0 else np.nan
    # Hedges g correction
    dfree = n0 + n1 - 2
    J = 1 - (3 / (4 * dfree - 1)) if dfree > 1 else 1
    g = d * J
else:
    d = np.nan
    g = np.nan

# Mixed effects model with random intercept for participant, if sufficient data
# Helps account for repeated measures per participant
mixed_result = None
mixed_p = np.nan
mixed_beta = np.nan
try:
    # Ensure there is more than one participant and both conditions present
    if sub["participant_id"].nunique() > 1 and sub["reader_view"].nunique() == 2:
        md = smf.mixedlm("reading_speed ~ reader_view", sub, groups=sub["participant_id"])
        mixed_result = md.fit(reml=False, method="lbfgs")
        mixed_beta = mixed_result.params.get("reader_view", np.nan)
        mixed_p = mixed_result.pvalues.get("reader_view", np.nan)
except Exception:
    mixed_result = None

# Save a compact JSON summary for later use
summary = {
    "speed_source": speed_source,
    "corr_total": float(corr_total),
    "corr_read": float(corr_read),
    "n_dyslexia": int(sub.shape[0]),
    "n_rv0": int(n0),
    "n_rv1": int(n1),
    "mean_rv0": float(mean0),
    "mean_rv1": float(mean1),
    "mean_diff": float(mean1 - mean0),
    "t_stat": float(t_stat) if np.isfinite(t_stat) else None,
    "p_val": float(p_val) if np.isfinite(p_val) else None,
    "hedges_g": float(g) if np.isfinite(g) else None,
    "mixed_beta": float(mixed_beta) if np.isfinite(mixed_beta) else None,
    "mixed_p": float(mixed_p) if np.isfinite(mixed_p) else None,
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
