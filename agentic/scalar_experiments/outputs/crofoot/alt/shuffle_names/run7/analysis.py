import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("crofoot.csv")

print("rows", len(df))
print(df.head())

# Map columns based on info.json descriptions
# Outcome: m_focal (1 if focal won)
# Focal group size: f_other
# Other group size: win
# Distances to home-range centers: m_other (focal), n_focal (other)

outcome = df["m_focal"].astype(int)

focal_size = df["f_other"]
other_size = df["win"]

focal_dist = df["m_other"]
other_dist = df["n_focal"]

rel_size = focal_size - other_size
rel_dist = other_dist - focal_dist  # positive => focal closer to its center

# Assemble modeling frame
model_df = pd.DataFrame({
    "win": outcome,
    "rel_size": rel_size,
    "rel_dist": rel_dist,
    "focal_size": focal_size,
    "other_size": other_size,
    "focal_dist": focal_dist,
    "other_dist": other_dist,
})

print("Outcome counts:\n", model_df["win"].value_counts())
print("rel_size summary:\n", model_df["rel_size"].describe())
print("rel_dist summary:\n", model_df["rel_dist"].describe())

# Logistic regression with relative size and relative distance
X = model_df[["rel_size", "rel_dist"]]
X = sm.add_constant(X)

logit = sm.Logit(model_df["win"], X)
result = logit.fit(disp=False)

print("\nLogit results (win ~ rel_size + rel_dist):")
print(result.summary())

# Odds ratios
params = result.params
conf = result.conf_int()

odds = np.exp(params)
conf_odds = np.exp(conf)

odds_table = pd.DataFrame({
    "odds_ratio": odds,
    "ci_low": conf_odds[0],
    "ci_high": conf_odds[1],
    "p_value": result.pvalues,
})
print("\nOdds ratios:")
print(odds_table)

# Simple models for each predictor separately
for var in ["rel_size", "rel_dist"]:
    Xs = sm.add_constant(model_df[[var]])
    r = sm.Logit(model_df["win"], Xs).fit(disp=False)
    print(f"\nLogit results (win ~ {var}):")
    print(r.summary())
    odds_var = np.exp(r.params)
    conf_var = np.exp(r.conf_int())
    print("Odds ratio:", odds_var[var], "CI", tuple(conf_var.loc[var]), "p", r.pvalues[var])
