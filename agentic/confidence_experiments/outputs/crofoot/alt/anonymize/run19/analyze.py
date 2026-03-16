import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Map columns using metadata descriptions
# feature4: outcome (1 focal won, 0 other won)
# feature5: distance of focal from center of its home range
# feature6: distance of other from center of its home range
# feature7: number of individuals in focal group
# feature8: number of individuals in other group

# Construct relative predictors
# Relative group size: focal size minus other size
# Relative location advantage: other distance minus focal distance (positive => focal closer to its center)
df = df.copy()
df["rel_size"] = df["feature7"] - df["feature8"]
df["rel_location"] = df["feature6"] - df["feature5"]

# Outcome
y = df["feature4"].astype(float)

# Predictors
X = df[["rel_size", "rel_location"]].astype(float)
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Compute odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf.columns = ["2.5%", "97.5%"]
odds = np.exp(params)
odds_ci = np.exp(conf)

output = {
    "n": int(len(df)),
    "pseudo_r2": float(result.prsquared),
    "params": params.to_dict(),
    "pvalues": result.pvalues.to_dict(),
    "odds_ratios": odds.to_dict(),
    "odds_ci": odds_ci.to_dict(),
}

# Also compute simple bivariate correlations for context (point-biserial)
# Using Pearson between predictors and outcome
corrs = {}
for col in ["rel_size", "rel_location"]:
    corrs[col] = float(np.corrcoef(df[col], y)[0, 1])
output["corrs"] = corrs

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
