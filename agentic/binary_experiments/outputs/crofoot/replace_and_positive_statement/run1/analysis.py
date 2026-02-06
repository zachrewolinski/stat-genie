import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path(__file__).with_name("crofoot.csv")

df = pd.read_csv(DATA_PATH)

# Relative group size (positive means focal larger)
df["rel_size"] = df["n_focal"] - df["n_other"]
# Contest location: positive means focal is closer to its home range center than the other group
# (other group's distance minus focal group's distance)
df["rel_loc"] = df["dist_other"] - df["dist_focal"]

# Standardize predictors for comparability
for col in ["rel_size", "rel_loc"]:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

X = sm.add_constant(df[["z_rel_size", "z_rel_loc"]])
y = df["win"]

model = sm.Logit(y, X).fit(disp=False)

# Compute predicted probabilities for typical changes
mean_row = X.mean().to_frame().T

# 1 SD increase in relative size
row_size_up = mean_row.copy()
row_size_up["z_rel_size"] = mean_row["z_rel_size"].iloc[0] + 1

# 1 SD increase in relative location advantage
row_loc_up = mean_row.copy()
row_loc_up["z_rel_loc"] = mean_row["z_rel_loc"].iloc[0] + 1

p_base = float(model.predict(mean_row))
p_size_up = float(model.predict(row_size_up))
p_loc_up = float(model.predict(row_loc_up))

print("Logit model: win ~ z_rel_size + z_rel_loc")
print(model.summary())
print("\nKey effects (z-scored predictors):")
print(model.params)
print("P-values:")
print(model.pvalues)
print("\nPredicted win probability at mean predictors:", round(p_base, 3))
print("Predicted win probability +1 SD rel_size:", round(p_size_up, 3))
print("Predicted win probability +1 SD rel_loc:", round(p_loc_up, 3))

# Simple descriptive checks
print("\nWin rate by relative size category:")
print(df.groupby(pd.cut(df["rel_size"], bins=[-10, -1, 0, 10]))["win"].mean())

print("\nWin rate by relative location advantage (rel_loc) sign:")
print(df.groupby(np.sign(df["rel_loc"]))["win"].mean())
