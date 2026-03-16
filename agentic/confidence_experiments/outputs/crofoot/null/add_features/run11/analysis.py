import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Keep needed columns, drop missing
cols = ["win", "n_focal", "n_other", "dist_focal", "dist_other"]
d = df[cols].copy().dropna()

# Derived variables
# Relative group size: positive means focal larger
# Relative location: positive means contest closer to focal group's center (other is farther from its center)
d["rel_size"] = d["n_focal"] - d["n_other"]
d["rel_location"] = d["dist_other"] - d["dist_focal"]

# Standard deviations for effect size interpretation
sd_rel_size = d["rel_size"].std(ddof=1)
sd_rel_location = d["rel_location"].std(ddof=1)

# Logistic regression
X = d[["rel_size", "rel_location"]]
X = sm.add_constant(X)

y = d["win"]

model = sm.GLM(y, X, family=sm.families.Binomial())
res = model.fit()

# Also fit univariate models for context
X_size = sm.add_constant(d[["rel_size"]])
res_size = sm.GLM(y, X_size, family=sm.families.Binomial()).fit()

X_loc = sm.add_constant(d[["rel_location"]])
res_loc = sm.GLM(y, X_loc, family=sm.families.Binomial()).fit()

# Summaries
print("N:", len(d))
print("rel_size mean/sd:", d["rel_size"].mean(), sd_rel_size)
print("rel_location mean/sd:", d["rel_location"].mean(), sd_rel_location)

print("\nFull model coefficients:")
print(res.params)
print("P-values:")
print(res.pvalues)
print("\nOdds ratios:")
print(np.exp(res.params))

# Odds ratios for 1 SD increase
or_size_1sd = float(np.exp(res.params["rel_size"] * sd_rel_size))
or_loc_1sd = float(np.exp(res.params["rel_location"] * sd_rel_location))
print("\nOR per 1 SD rel_size:", or_size_1sd)
print("OR per 1 SD rel_location:", or_loc_1sd)

print("\nUnivariate models p-values:")
print("rel_size:", res_size.pvalues["rel_size"])
print("rel_location:", res_loc.pvalues["rel_location"])

# Compute predicted win probability difference for typical changes
# Use mean of predictors
mean_row = pd.DataFrame({
    "const": [1.0],
    "rel_size": [d["rel_size"].mean()],
    "rel_location": [d["rel_location"].mean()],
})

# 1 SD increase in each predictor
row_size_up = mean_row.copy()
row_size_up["rel_size"] += sd_rel_size
row_loc_up = mean_row.copy()
row_loc_up["rel_location"] += sd_rel_location

p_base = float(res.predict(mean_row)[0])
p_size_up = float(res.predict(row_size_up)[0])
p_loc_up = float(res.predict(row_loc_up)[0])

print("\nPredicted win probability at mean predictors:", p_base)
print("Predicted win probability +1 SD rel_size:", p_size_up)
print("Predicted win probability +1 SD rel_location:", p_loc_up)
