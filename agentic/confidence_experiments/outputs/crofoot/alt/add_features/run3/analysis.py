import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("crofoot.csv")

# Derived variables

df["rel_size"] = df["n_focal"] - df["n_other"]
# Positive rel_loc means the contest is closer to the focal group's home range center

df["rel_loc"] = df["dist_other"] - df["dist_focal"]

df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

# Logistic regression with continuous rel_size and rel_loc
model_cont = smf.logit("win ~ rel_size + rel_loc", data=df).fit(disp=False)

# Logistic regression with binary location (focal closer) for interpretability
model_bin = smf.logit("win ~ rel_size + focal_closer", data=df).fit(disp=False)

# Also check univariate models
model_size = smf.logit("win ~ rel_size", data=df).fit(disp=False)
model_loc = smf.logit("win ~ rel_loc", data=df).fit(disp=False)

# Output summaries
print("N:", len(df))
print("\nModel: win ~ rel_size + rel_loc")
print(model_cont.summary())
print("\nModel: win ~ rel_size + focal_closer")
print(model_bin.summary())
print("\nModel: win ~ rel_size")
print(model_size.summary())
print("\nModel: win ~ rel_loc")
print(model_loc.summary())


def extract(model, var):
    return {
        "coef": model.params[var],
        "se": model.bse[var],
        "p": model.pvalues[var],
        "odds_ratio": float(np.exp(model.params[var])),
    }


results = {
    "model_cont": {
        "rel_size": extract(model_cont, "rel_size"),
        "rel_loc": extract(model_cont, "rel_loc"),
        "intercept": extract(model_cont, "Intercept"),
        "pseudo_r2": model_cont.prsquared,
    },
    "model_bin": {
        "rel_size": extract(model_bin, "rel_size"),
        "focal_closer": extract(model_bin, "focal_closer"),
        "intercept": extract(model_bin, "Intercept"),
        "pseudo_r2": model_bin.prsquared,
    },
    "model_size": {
        "rel_size": extract(model_size, "rel_size"),
        "intercept": extract(model_size, "Intercept"),
        "pseudo_r2": model_size.prsquared,
    },
    "model_loc": {
        "rel_loc": extract(model_loc, "rel_loc"),
        "intercept": extract(model_loc, "Intercept"),
        "pseudo_r2": model_loc.prsquared,
    },
}

print("\nKey stats:")
for k, v in results.items():
    print(k, v)

# Save results to JSON for later use
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
