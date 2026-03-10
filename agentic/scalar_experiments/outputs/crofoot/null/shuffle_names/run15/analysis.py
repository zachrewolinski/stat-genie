import pandas as pd
import statsmodels.api as sm

DATA_PATH = "crofoot.csv"

df = pd.read_csv(DATA_PATH)

# Outcome: focal win (binary)
# Predictors: relative group size and relative contest location
# Based on metadata: f_other = focal group size, win = other group size
# m_other = distance of focal group from its home range center
# n_focal = distance of other group from its home range center

df = df.copy()

df["rel_group_size"] = df["f_other"] - df["win"]
# Negative values indicate contest closer to focal group's center (advantage for focal)
# We'll keep sign as focal distance minus other distance

df["rel_location"] = df["m_other"] - df["n_focal"]

# Logistic regression
X = df[["rel_group_size", "rel_location"]]
X = sm.add_constant(X)
y = df["m_focal"]

model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Also compute pseudo-R2 and odds ratios
params = result.params
conf = result.conf_int()
conf.columns = ["2.5%", "97.5%"]

or_df = pd.DataFrame({
    "odds_ratio": params.apply(lambda x: float(pd.np.exp(x))),
    "p_value": result.pvalues,
    "ci_lower": conf["2.5%"].apply(lambda x: float(pd.np.exp(x))),
    "ci_upper": conf["97.5%"].apply(lambda x: float(pd.np.exp(x))),
})

print("\nOdds ratios (exp(coef)):")
print(or_df)

print("\nPseudo R2 (McFadden):", result.prsquared)

# Simple correlations for context
print("\nPoint-biserial correlations with outcome:")
print(df[["rel_group_size", "rel_location", "m_focal"]].corr().loc[["rel_group_size", "rel_location"], "m_focal"])
