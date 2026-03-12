import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "crofoot.csv"

df = pd.read_csv(DATA_PATH)

# Outcome
win = df["feature4"].astype(int)

# Relative group size (focal - other)
rel_size = df["feature7"] - df["feature8"]

# Relative location advantage: other distance - focal distance
# Positive means focal is closer to its own center than the other group is to its center.
rel_location = df["feature6"] - df["feature5"]

X = pd.DataFrame({"rel_size": rel_size, "rel_location": rel_location})
X = sm.add_constant(X)

model = sm.Logit(win, X)
result = model.fit(disp=False)

# Also fit model with standardized predictors for effect sizes
X_std = pd.DataFrame({
    "rel_size": (rel_size - rel_size.mean()) / rel_size.std(ddof=0),
    "rel_location": (rel_location - rel_location.mean()) / rel_location.std(ddof=0),
})
X_std = sm.add_constant(X_std)
result_std = sm.Logit(win, X_std).fit(disp=False)

# Odds ratios
odds_ratios = np.exp(result.params)

output = {
    "n": int(len(df)),
    "coef": result.params.to_dict(),
    "pvalues": result.pvalues.to_dict(),
    "odds_ratios": odds_ratios.to_dict(),
    "coef_std": result_std.params.to_dict(),
    "pvalues_std": result_std.pvalues.to_dict(),
    "pseudo_r2": result.prsquared,
}

print(json.dumps(output, indent=2))
