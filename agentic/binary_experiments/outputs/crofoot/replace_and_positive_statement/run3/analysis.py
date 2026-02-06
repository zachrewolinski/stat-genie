import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Feature engineering: relative group size and contest location
_df["size_diff"] = _df["n_focal"] - _df["n_other"]
# Positive loc_diff means focal is farther from its home-range center than the other group
_df["loc_diff"] = _df["dist_focal"] - _df["dist_other"]

# Logistic regression: win ~ size_diff + loc_diff
X = sm.add_constant(_df[["size_diff", "loc_diff"]])
model = sm.Logit(_df["win"], X).fit(disp=False)

# Summaries for interpretation
params = model.params
pvals = model.pvalues

# Odds ratio for a 100m shift in loc_diff
or_loc_100m = float(np.exp(params["loc_diff"] * 100))

# Simple descriptive win rates by sign of predictors
_df["size_diff_pos"] = _df["size_diff"] > 0
_df["loc_diff_pos"] = _df["loc_diff"] > 0
win_by_size = _df.groupby("size_diff_pos")["win"].mean()
win_by_loc = _df.groupby("loc_diff_pos")["win"].mean()

print("Logit model: win ~ size_diff + loc_diff")
print(model.summary())
print("\nCoefficients:")
print(params)
print("\nP-values:")
print(pvals)
print(f"\nOdds ratio for +100m loc_diff (focal farther from center): {or_loc_100m:.3f}")
print("\nWin rate by size_diff > 0:")
print(win_by_size)
print("\nWin rate by loc_diff > 0:")
print(win_by_loc)
