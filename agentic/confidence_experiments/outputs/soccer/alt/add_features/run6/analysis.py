import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute skin tone as mean of two raters (0 to 1 scale)
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin=skin)

# Filter to rows with skin and games > 0
analysis_df = df.loc[df["skin"].notna() & (df["games"] > 0)].copy()

# Define light and dark extremes
analysis_df["skin_group"] = np.where(
    analysis_df["skin"] <= 0.25, "light",
    np.where(analysis_df["skin"] >= 0.75, "dark", "mid")
)

# Summary counts
summary = (
    analysis_df.groupby("skin_group")[["redCards", "games"]]
    .sum()
    .assign(rate_per_100=lambda d: 100 * d["redCards"] / d["games"])
)

# Poisson GLM for dark vs light with log(games) offset
extreme_df = analysis_df.loc[analysis_df["skin_group"].isin(["light", "dark"])].copy()
extreme_df["dark"] = (extreme_df["skin_group"] == "dark").astype(int)

# Add intercept
X = sm.add_constant(extreme_df[["dark"]])
y = extreme_df["redCards"]
offset = np.log(extreme_df["games"])

model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
res = model.fit(cov_type="cluster", cov_kwds={"groups": extreme_df["playerShort"]})

beta_dark = res.params["dark"]
se_dark = res.bse["dark"]
p_dark = res.pvalues["dark"]
irr_dark = float(np.exp(beta_dark))

# Continuous skin tone model (all with skin)
X2 = sm.add_constant(analysis_df[["skin"]])
y2 = analysis_df["redCards"]
offset2 = np.log(analysis_df["games"])
model2 = sm.GLM(y2, X2, family=sm.families.Poisson(), offset=offset2)
res2 = model2.fit(cov_type="cluster", cov_kwds={"groups": analysis_df["playerShort"]})

beta_skin = res2.params["skin"]
se_skin = res2.bse["skin"]
p_skin = res2.pvalues["skin"]
irr_skin = float(np.exp(beta_skin))

# Predicted rates per 100 games for light vs dark using model 1
# rate = exp(const + beta*dark) * games; with games=1 -> per game
const = res.params["const"]
rate_light = np.exp(const) * 100
rate_dark = np.exp(const + beta_dark) * 100

out = {
    "n_rows_total": int(len(df)),
    "n_rows_with_skin": int(len(analysis_df)),
    "summary_rates": summary.reset_index().to_dict(orient="records"),
    "extreme_n": int(len(extreme_df)),
    "poisson_dark_vs_light": {
        "beta_dark": float(beta_dark),
        "se_dark": float(se_dark),
        "p_dark": float(p_dark),
        "irr_dark": irr_dark,
        "rate_per_100_light_model": float(rate_light),
        "rate_per_100_dark_model": float(rate_dark),
    },
    "poisson_continuous_skin": {
        "beta_skin": float(beta_skin),
        "se_skin": float(se_skin),
        "p_skin": float(p_skin),
        "irr_per_1unit_skin": irr_skin,
    },
}

print(json.dumps(out, indent=2))
