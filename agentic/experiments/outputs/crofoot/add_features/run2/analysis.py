import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Feature engineering for research question
# Relative group size (positive means focal larger)
df["rel_size"] = df["n_focal"] - df["n_other"]
# Relative location: positive means contest closer to focal home range center
# (other distance minus focal distance)
df["rel_dist"] = df["dist_other"] - df["dist_focal"]

# Basic descriptive stats
print("Rows:", len(df))
print("Win rate:", df["win"].mean())
print("Rel_size summary:")
print(df["rel_size"].describe())
print("Rel_dist summary:")
print(df["rel_dist"].describe())

# Logistic regression: win ~ rel_size + rel_dist
X = df[["rel_size", "rel_dist"]].copy()
X = sm.add_constant(X)
y = df["win"]

logit_model = sm.Logit(y, X).fit(disp=False)
print("\nLogit results:")
print(logit_model.summary())

# Odds ratios for interpretability
params = logit_model.params
conf = logit_model.conf_int()
conf.columns = ["2.5%", "97.5%"]

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print("\nOdds ratios:")
print(odds_ratios)
print("\nOdds ratio 95% CI:")
print(conf_or)

# Simple stratified win rates for intuition
# Quartiles for rel_dist (location advantage)
quartiles = pd.qcut(df["rel_dist"], 4, labels=False)
print("\nWin rate by rel_dist quartile (0=most other-advantaged, 3=most focal-advantaged):")
print(df.groupby(quartiles)["win"].mean())

# Group size advantage categories
size_cats = pd.cut(df["rel_size"], bins=[-10, -2, -1, 0, 1, 2, 10], right=True)
print("\nWin rate by rel_size bin:")
print(df.groupby(size_cats)["win"].mean())
