import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Derived predictors
# Relative group size (focal - other)
df["size_diff"] = df["n_focal"] - df["n_other"]

# Location advantage: positive means focal is closer to its center than other is to its center
# (other distance - focal distance)
df["loc_adv"] = df["dist_other"] - df["dist_focal"]

# Logistic regression
X = df[["size_diff", "loc_adv"]].copy()
X = sm.add_constant(X)
y = df["win"]

model = sm.Logit(y, X).fit(disp=False)

# Standardized predictors for effect size comparability
X_std = df[["size_diff", "loc_adv"]].apply(lambda s: (s - s.mean())/s.std(ddof=0))
X_std = sm.add_constant(X_std)
model_std = sm.Logit(y, X_std).fit(disp=False)

# Simple descriptive stats
win_rate = df["win"].mean()

# Grouped win rates by size advantage and location advantage
size_adv_win = df[df["size_diff"] > 0]["win"].mean()
size_disadv_win = df[df["size_diff"] < 0]["win"].mean()
size_tie_win = df[df["size_diff"] == 0]["win"].mean()

loc_adv_win = df[df["loc_adv"] > 0]["win"].mean()
loc_disadv_win = df[df["loc_adv"] < 0]["win"].mean()
loc_tie_win = df[df["loc_adv"] == 0]["win"].mean()

print("n_rows", len(df))
print("win_rate", win_rate)
print("size_adv_win", size_adv_win, "size_disadv_win", size_disadv_win, "size_tie_win", size_tie_win)
print("loc_adv_win", loc_adv_win, "loc_disadv_win", loc_disadv_win, "loc_tie_win", loc_tie_win)

print("\nLogit coefficients (raw):")
print(model.summary2().tables[1])

print("\nLogit coefficients (standardized):")
print(model_std.summary2().tables[1])
