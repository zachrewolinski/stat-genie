import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# Load data
csv_path = Path("crofoot.csv")
df = pd.read_csv(csv_path)

# Rename columns for clarity
cols = {
    "feature1": "focal_group_id",
    "feature2": "other_group_id",
    "feature3": "dyad_id",
    "feature4": "focal_win",
    "feature5": "focal_dist_center",
    "feature6": "other_dist_center",
    "feature7": "focal_n",
    "feature8": "other_n",
    "feature9": "focal_males",
    "feature10": "other_males",
    "feature11": "focal_females",
    "feature12": "other_females",
}

df = df.rename(columns=cols)

# Key predictors based on research question
# Relative group size (focal - other)
df["rel_size"] = df["focal_n"] - df["other_n"]

# Location advantage: negative if focal is closer to its center than other is to its center
# (i.e., focal has home-range advantage), positive if the reverse
# Also capture absolute difference in distance from own center

df["delta_home"] = df["focal_dist_center"] - df["other_dist_center"]

# Standardize continuous predictors for stability
for col in ["rel_size", "delta_home"]:
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        df[col + "_z"] = 0.0
    else:
        df[col + "_z"] = (df[col] - mean) / std

# Prepare design matrix
X = df[["rel_size_z", "delta_home_z"]]
X = sm.add_constant(X)
y = df["focal_win"]

# Fit logistic regression (GLM binomial)
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Extract coefficients and p-values
coef = result.params
pvalues = result.pvalues

rel_size_beta = float(coef.get("rel_size_z", 0.0))
rel_size_p = float(pvalues.get("rel_size_z", 1.0))

loc_beta = float(coef.get("delta_home_z", 0.0))
loc_p = float(pvalues.get("delta_home_z", 1.0))

# Summarize evidence that predictors influence win probability
# We'll convert effect size + significance into a 0-1 evidence score.

import math

def evidence_from_effect(beta: float, p: float) -> float:
    # Noisy but monotonic mapping: larger |beta| and smaller p -> higher score
    abs_beta = abs(beta)
    # cap beta effect
    effect_component = math.tanh(abs_beta)
    # convert p-value (capped at 1e-6) to 0-1, with 0.05 ~ mid-strength
    p = max(min(p, 1.0), 1e-6)
    sig_component = 1.0 - min(-math.log10(p) / 6.0, 1.0)  # smaller p -> closer to 0
    sig_component = 1.0 - sig_component  # invert so smaller p -> higher score
    # combine
    return max(0.0, min(1.0, 0.5 * effect_component + 0.5 * sig_component))

rel_evid = evidence_from_effect(rel_size_beta, rel_size_p)
loc_evid = evidence_from_effect(loc_beta, loc_p)

# Overall evidence that BOTH relative size and location influence win probability.
# Use average of the two evidence scores.
overall_evidence = (rel_evid + loc_evid) / 2.0

# Map 0-1 evidence to Likert scale -100..100 where
# 0 ~ strong "No influence", 0.5 ~ weak/ambiguous, 1 ~ very strong "Yes influence".
# We'll center neutral (0 on Likert) at evidence 0.5.
likert = int(round((overall_evidence - 0.5) * 200))

# Clip to [-100, 100]
likert = max(-100, min(100, likert))

# Write scalar to conclusion.txt
out_path = Path("conclusion.txt")
out_path.write_text(str(likert), encoding="utf-8")

