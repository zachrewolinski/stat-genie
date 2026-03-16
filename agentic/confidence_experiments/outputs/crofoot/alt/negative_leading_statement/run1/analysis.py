import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Derived predictors
# Relative group size advantage for focal group
# Positive means focal group larger than other
# Location advantage: positive means contest closer to focal group's home-range center
# (other group is farther from its own center)
df = df.copy()
df["size_diff"] = df["n_focal"] - df["n_other"]
df["location_adv"] = df["dist_other"] - df["dist_focal"]

# Fit logistic regression
model = smf.logit("win ~ size_diff + location_adv", data=df).fit(disp=False)

# Also fit single-predictor models for context
model_size = smf.logit("win ~ size_diff", data=df).fit(disp=False)
model_loc = smf.logit("win ~ location_adv", data=df).fit(disp=False)

# Extract results
results = {
    "n": int(df.shape[0]),
    "model": {
        "coef": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "conf_int": {k: v for k, v in model.conf_int().iterrows()},
        "odds_ratio": np.exp(model.params).to_dict(),
    },
    "model_size": {
        "coef": model_size.params.to_dict(),
        "pvalues": model_size.pvalues.to_dict(),
        "odds_ratio": np.exp(model_size.params).to_dict(),
    },
    "model_loc": {
        "coef": model_loc.params.to_dict(),
        "pvalues": model_loc.pvalues.to_dict(),
        "odds_ratio": np.exp(model_loc.params).to_dict(),
    },
    "summary": str(model.summary()),
}

print(json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))
