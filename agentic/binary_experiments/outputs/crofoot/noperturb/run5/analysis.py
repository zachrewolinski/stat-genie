import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Construct predictors
_df["rel_size"] = _df["n_focal"] - _df["n_other"]
# Positive loc_adv means contest is closer to focal group's home-range center
_df["loc_adv"] = _df["dist_other"] - _df["dist_focal"]

# Logistic regression: win ~ rel_size + loc_adv
X = _df[["rel_size", "loc_adv"]]
X = sm.add_constant(X)
y = _df["win"]
model = sm.Logit(y, X).fit(disp=False)

# Compute odds ratios and p-values
params = model.params
conf = model.conf_int()
conf.columns = ["2.5%", "97.5%"]
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print("Logit model: win ~ rel_size + loc_adv")
print(model.summary())
print("\nOdds ratios:")
print(pd.DataFrame({"OR": odds_ratios, "p": model.pvalues}))
print("\nOR 95% CI:")
print(conf_or)

# Simple descriptive check: win rate by sign of location advantage
_df["loc_side"] = np.where(_df["loc_adv"] >= 0, "closer_to_focal", "closer_to_other")
win_rates = _df.groupby("loc_side")["win"].mean()
print("\nWin rate by contest location side:")
print(win_rates)

# Correlations (descriptive)
print("\nCorrelations with win:")
print({
    "rel_size": np.corrcoef(_df["win"], _df["rel_size"])[0, 1],
    "loc_adv": np.corrcoef(_df["win"], _df["loc_adv"])[0, 1],
})
