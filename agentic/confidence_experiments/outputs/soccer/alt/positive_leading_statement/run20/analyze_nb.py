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
player["log_games"] = np.log(player["games"])

# Negative Binomial regression with offset
X = sm.add_constant(player["skin"])
model_nb = sm.NegativeBinomial(player["redCards"], X, offset=player["log_games"])
res_nb = model_nb.fit(disp=False)
coef = res_nb.params["skin"]
pval = res_nb.pvalues["skin"]
rr_0_1 = float(np.exp(coef * 0.1))

print("NB CONTINUOUS SKIN")
print(f"coef_skin: {coef}")
print(f"p_skin: {pval}")
print(f"rr_per_0.1_skin: {rr_0_1}")
print(f"alpha: {res_nb.params['alpha']}")

# Dark vs light
player_dl = player[(player["skin"] <= 0.4) | (player["skin"] >= 0.6)].copy()
player_dl["dark"] = (player_dl["skin"] >= 0.6).astype(int)
X_dl = sm.add_constant(player_dl["dark"])
model_nb_dl = sm.NegativeBinomial(player_dl["redCards"], X_dl, offset=np.log(player_dl["games"]))
res_nb_dl = model_nb_dl.fit(disp=False)
coef_dl = res_nb_dl.params["dark"]
pval_dl = res_nb_dl.pvalues["dark"]
rr_dl = float(np.exp(coef_dl))

print("\nNB DARK VS LIGHT")
print(f"coef_dark: {coef_dl}")
print(f"p_dark: {pval_dl}")
print(f"rr_dark_vs_light: {rr_dl}")
print(f"alpha: {res_nb_dl.params['alpha']}")

