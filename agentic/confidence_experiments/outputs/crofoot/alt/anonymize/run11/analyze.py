import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "crofoot.csv"

df = pd.read_csv(DATA_PATH)

# Rename for clarity
cols = df.columns.tolist()
# feature4 is outcome, feature5/6 distances, feature7/8 group sizes

# Derived variables
# Relative group size: focal size minus other size
rel_size = df["feature7"] - df["feature8"]
# Also ratio for sensitivity
rel_size_ratio = df["feature7"] / df["feature8"]

# Contest location proxy: difference in distance to own home-range center
# Positive means focal farther from its center than other is from its center
loc_diff = df["feature5"] - df["feature6"]
# Also relative location: focal closer than other to its own center
focal_closer = (df["feature5"] < df["feature6"]).astype(int)

outcome = df["feature4"]

# Prepare logistic regression with rel_size and location difference
X = pd.DataFrame({
    "rel_size": rel_size,
    "loc_diff": loc_diff,
})
X = sm.add_constant(X)
model = sm.Logit(outcome, X).fit(disp=False)

# Alternative model using ratio + binary location
X2 = pd.DataFrame({
    "rel_size_ratio": rel_size_ratio,
    "focal_closer": focal_closer,
})
X2 = sm.add_constant(X2)
model2 = sm.Logit(outcome, X2).fit(disp=False)

# Simple bivariate tests
# Correlation (point-biserial) for rel_size and loc_diff
corr_rel = np.corrcoef(rel_size, outcome)[0,1]
corr_loc = np.corrcoef(loc_diff, outcome)[0,1]

# Summaries
summary = {
    "n": int(len(df)),
    "model1_params": model.params.to_dict(),
    "model1_pvalues": model.pvalues.to_dict(),
    "model1_confint": {k: [float(v) for v in ci] for k, ci in model.conf_int().to_dict("index").items()},
    "model1_pseudo_r2": float(model.prsquared),
    "model2_params": model2.params.to_dict(),
    "model2_pvalues": model2.pvalues.to_dict(),
    "model2_confint": {k: [float(v) for v in ci] for k, ci in model2.conf_int().to_dict("index").items()},
    "model2_pseudo_r2": float(model2.prsquared),
    "corr_rel_size_outcome": float(corr_rel),
    "corr_loc_diff_outcome": float(corr_loc),
    "win_rate": float(outcome.mean()),
}

print(json.dumps(summary, indent=2))
