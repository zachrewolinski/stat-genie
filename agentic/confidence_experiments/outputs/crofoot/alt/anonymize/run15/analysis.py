import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Rename variables for clarity
win = _df["feature4"].astype(int)
rel_size = _df["feature7"] - _df["feature8"]  # focal group size minus other group size
rel_location = _df["feature5"] - _df["feature6"]  # focal distance to its center minus other's distance

# Assemble analysis dataframe
_df2 = pd.DataFrame({
    "win": win,
    "rel_size": rel_size,
    "rel_location": rel_location,
})

# Standardize predictors for comparable effect sizes
_df2["rel_size_z"] = (_df2["rel_size"] - _df2["rel_size"].mean()) / _df2["rel_size"].std(ddof=0)
_df2["rel_location_z"] = (_df2["rel_location"] - _df2["rel_location"].mean()) / _df2["rel_location"].std(ddof=0)

# Logistic regression: win ~ rel_size + rel_location
X = sm.add_constant(_df2[["rel_size", "rel_location"]])
model = sm.GLM(_df2["win"], X, family=sm.families.Binomial())
result = model.fit()

# Standardized version
Xz = sm.add_constant(_df2[["rel_size_z", "rel_location_z"]])
model_z = sm.GLM(_df2["win"], Xz, family=sm.families.Binomial())
result_z = model_z.fit()

# Marginal win rates by advantage categories
size_adv = pd.cut(_df2["rel_size"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["smaller", "equal", "larger"])
loc_adv = pd.cut(_df2["rel_location"], bins=[-np.inf, -1e-9, 1e-9, np.inf], labels=["closer", "tie", "farther"])

size_rates = _df2.groupby(size_adv, observed=True)["win"].agg(["mean", "count"]).reset_index()
loc_rates = _df2.groupby(loc_adv, observed=True)["win"].agg(["mean", "count"]).reset_index()

# Print results
print("N:", len(_df2))
print("Win rate overall:", _df2["win"].mean())
print("\nLogistic regression (unstandardized):")
print(result.summary())
print("\nLogistic regression (standardized):")
print(result_z.summary())

print("\nWin rates by relative size category:")
print(size_rates)

print("\nWin rates by relative location category:")
print(loc_rates)

# Odds ratios with 95% CI for unstandardized predictors
params = result.params
conf = result.conf_int()
OR = np.exp(params)
OR_ci = np.exp(conf)
print("\nOdds ratios (unstandardized) and 95% CI:")
for name in params.index:
    print(name, OR[name], OR_ci.loc[name, 0], OR_ci.loc[name, 1])
