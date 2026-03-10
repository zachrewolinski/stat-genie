import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("soccer.csv")
df["skin"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
player = (
    df.groupby("playerShort", as_index=False)
      .agg(
          skin=("skin", "mean"),
          games=("games", "sum"),
          redCards=("redCards", "sum"),
      )
)
player = player.dropna(subset=["skin"])
player = player[player["games"] > 0]

# threshold at 0.5
player["dark"] = (player["skin"] > 0.5).astype(int)

# totals
light = player[player["dark"] == 0]
dark = player[player["dark"] == 1]
light_rate = light["redCards"].sum() / light["games"].sum()
dark_rate = dark["redCards"].sum() / dark["games"].sum()
rr = dark_rate / light_rate

# poisson regression
X = sm.add_constant(player["dark"])
res = sm.GLM(player["redCards"], X, family=sm.families.Poisson(), offset=np.log(player["games"])).fit(cov_type="HC0")
coef = res.params["dark"]
pval = res.pvalues["dark"]

print("n_light", len(light))
print("n_dark", len(dark))
print("light_rate", light_rate)
print("dark_rate", dark_rate)
print("rr", rr)
print("coef", coef)
print("p", pval)

