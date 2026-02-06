import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Relative group size: focal minus other
_df["size_diff"] = _df["n_focal"] - _df["n_other"]

# Contest location advantage: positive if focal closer to its home-range center
_df["loc_adv"] = _df["dist_other"] - _df["dist_focal"]

# Logistic regression
_X = _df[["size_diff", "loc_adv"]]
_X = sm.add_constant(_X)
_y = _df["win"]

_model = sm.Logit(_y, _X).fit(disp=False)

print(_model.summary())

# Extract key results
_params = _model.params
_pvals = _model.pvalues

print("\nKey effects:")
for _var in ["size_diff", "loc_adv"]:
    print(f"{_var}: coef={_params[_var]:.4f}, p-value={_pvals[_var]:.4f}")

# Simple effect sizes: odds ratios for 1-unit change
_odds_ratios = np.exp(_params[["size_diff", "loc_adv"]])
print("\nOdds ratios (per 1-unit increase):")
print(_odds_ratios)
