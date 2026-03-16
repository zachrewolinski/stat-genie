import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("soccer.csv")

# Compute mean skin tone rating (0=very light, 1=very dark)
_df["skin_tone"] = _df[["feature18", "feature19"]].mean(axis=1)

# Keep rows with needed data
_df = _df.dropna(subset=["skin_tone", "feature9", "feature16"])
_df = _df[_df["feature9"] > 0]

# Light vs dark groups (extremes)
light_mask = _df["skin_tone"] <= 0.25
_dark_mask = _df["skin_tone"] >= 0.75
_df_ld = _df[light_mask | _dark_mask].copy()
_df_ld["dark"] = (_df_ld["skin_tone"] >= 0.75).astype(int)

# Aggregate rates
grp = _df_ld.groupby("dark").agg(
    red_cards=("feature16", "sum"),
    games=("feature9", "sum"),
    dyads=("feature16", "size"),
)

rate_light = grp.loc[0, "red_cards"] / grp.loc[0, "games"]
rate_dark = grp.loc[1, "red_cards"] / grp.loc[1, "games"]
rate_ratio = rate_dark / rate_light

# Poisson regression with offset for games played
X = sm.add_constant(_df_ld[["dark"]])
model = sm.GLM(
    _df_ld["feature16"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(_df_ld["feature9"]),
)
res = model.fit(cov_type="HC1")
coef = res.params["dark"]
rr = float(np.exp(coef))
ci = np.exp(res.conf_int().loc["dark"]).tolist()

# Continuous skin tone model
Xc = sm.add_constant(_df[["skin_tone"]])
model_c = sm.GLM(
    _df["feature16"],
    Xc,
    family=sm.families.Poisson(),
    offset=np.log(_df["feature9"]),
)
res_c = model_c.fit(cov_type="HC1")
coef_c = res_c.params["skin_tone"]
rr_c = float(np.exp(coef_c))
ci_c = np.exp(res_c.conf_int().loc["skin_tone"]).tolist()

print("Rows total:", len(_df))
print("Rows light/dark:", len(_df_ld))
print("Light rate per game:", rate_light)
print("Dark rate per game:", rate_dark)
print("Rate ratio dark/light:", rate_ratio)
print("Poisson RR (dark vs light):", rr)
print("Poisson RR 95% CI:", ci)
print("Poisson p-value:", res.pvalues["dark"])
print("Continuous RR per 1.0 skin tone:", rr_c)
print("Continuous 95% CI:", ci_c)
print("Continuous p-value:", res_c.pvalues["skin_tone"])
