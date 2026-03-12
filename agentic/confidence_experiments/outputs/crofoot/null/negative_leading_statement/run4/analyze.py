import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Derived variables
# Relative group size (positive means focal larger)
df["rel_size"] = df["n_focal"] - df["n_other"]
# Relative location: positive means focal is closer to its home-range center than the other group
# (i.e., contest is relatively closer to focal's center)
df["rel_loc"] = df["dist_other"] - df["dist_focal"]

results = {}

# Model 1: relative size + relative location
X1 = sm.add_constant(df[["rel_size", "rel_loc"]])
model1 = sm.GLM(df["win"], X1, family=sm.families.Binomial()).fit()
results["model1"] = model1

# Model 2: absolute group sizes and distances
X2 = sm.add_constant(df[["n_focal", "n_other", "dist_focal", "dist_other"]])
model2 = sm.GLM(df["win"], X2, family=sm.families.Binomial()).fit()
results["model2"] = model2

# Model 3: relative size only
X3 = sm.add_constant(df[["rel_size"]])
model3 = sm.GLM(df["win"], X3, family=sm.families.Binomial()).fit()
results["model3"] = model3

# Model 4: relative location only
X4 = sm.add_constant(df[["rel_loc"]])
model4 = sm.GLM(df["win"], X4, family=sm.families.Binomial()).fit()
results["model4"] = model4

# Summaries
print("N rows:", len(df))
print("Win rate:", df["win"].mean())
print("\nModel 1: win ~ rel_size + rel_loc")
print(model1.summary())
print("\nModel 2: win ~ n_focal + n_other + dist_focal + dist_other")
print(model2.summary())
print("\nModel 3: win ~ rel_size")
print(model3.summary())
print("\nModel 4: win ~ rel_loc")
print(model4.summary())

# Odds ratios and confidence intervals for model1
params = model1.params
conf = model1.conf_int()
conf.columns = ["2.5%", "97.5%"]

import numpy as np
or_table = pd.DataFrame({
    "coef": params,
    "odds_ratio": np.exp(params),
})
conf_or = np.exp(conf)

print("\nModel 1 Odds Ratios (95% CI):")
print(pd.concat([or_table[["odds_ratio"]], conf_or], axis=1))

# Simple descriptive: mean win by relative size sign
sign_bins = pd.cut(df["rel_size"], bins=[-10, -1, 0, 10], labels=["focal smaller", "equal", "focal larger"], include_lowest=True)
print("\nWin rate by relative size category:")
print(df.groupby(sign_bins)["win"].mean())

# For location advantage: quartiles of rel_loc
loc_bins = pd.qcut(df["rel_loc"], q=4, duplicates='drop')
print("\nWin rate by relative location quartile:")
print(df.groupby(loc_bins)["win"].mean())
