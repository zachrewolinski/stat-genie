import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "reading.csv"
df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Use feature5 as reading time minus scrolling (ms). Avoid zero/negative times.
df = df.copy()
df["reading_time_ms"] = df["feature5"]

# Compute reading speed in words per minute
# Avoid divide by zero
valid = df["reading_time_ms"] > 0

# Words on page
words = df.loc[valid, "feature7"].astype(float)
time_min = df.loc[valid, "reading_time_ms"].astype(float) / 60000.0
speed_wpm = words / time_min

df = df.loc[valid].copy()
df["speed_wpm"] = speed_wpm

# Filter to dyslexic readers (feature17 == 1)
dys = df[df["feature17"] == 1].copy()

# If too few rows, fall back to using feature12 (>=1)
if dys.shape[0] < 30:
    dys = df[df["feature12"] >= 1].copy()

# Drop extreme outliers to stabilize estimates (1st-99th percentile)
low, high = dys["speed_wpm"].quantile([0.01, 0.99])
dys = dys[(dys["speed_wpm"] >= low) & (dys["speed_wpm"] <= high)].copy()

# Summary stats
summary = dys.groupby("feature3")["speed_wpm"].agg(["count", "mean", "std"]).reset_index()

# Simple difference in means
rv_on = dys[dys["feature3"] == 1]["speed_wpm"]
rv_off = dys[dys["feature3"] == 0]["speed_wpm"]

# Welch t-test using statsmodels
from statsmodels.stats.weightstats import ttest_ind

t_stat, p_val, _ = ttest_ind(rv_on, rv_off, usevar="unequal")

# Regression controlling for key covariates
# Use log speed to reduce skew
# Categorical covariates: device, language, education, native English, gender

dys = dys.copy()
dys["log_speed"] = np.log(dys["speed_wpm"])

# Build formula
formula = (
    "log_speed ~ feature3 + feature7 + feature10 + feature19 + "
    "C(feature11) + C(feature13) + C(feature15) + C(feature18) + C(feature14)"
)

model = smf.ols(formula=formula, data=dys).fit()

coef = model.params.get("feature3", np.nan)
p_val_reg = model.pvalues.get("feature3", np.nan)

# Save key results to a small text file for inspection (optional)
with open("analysis_results.txt", "w") as f:
    f.write("Dyslexic subset size: {}\n".format(dys.shape[0]))
    f.write("Group summary (feature3: 0=off, 1=on)\n")
    f.write(summary.to_string(index=False))
    f.write("\n\nWelch t-test: t={:.4f}, p={:.6f}\n".format(t_stat, p_val))
    f.write("Regression coef(feature3)={:.6f}, p={:.6f}\n".format(coef, p_val_reg))

# Print to stdout for quick check
print("Dyslexic subset size:", dys.shape[0])
print(summary)
print("Welch t-test p:", p_val)
print("Regression coef(feature3):", coef, "p:", p_val_reg)
