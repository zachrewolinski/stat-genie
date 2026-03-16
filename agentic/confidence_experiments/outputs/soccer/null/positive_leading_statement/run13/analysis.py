import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute skin tone average; values are normalized 0-1 in 0.25 steps
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Filter to rows with skin tone and positive games
analysis_df = df.loc[df["skin_tone"].notna() & (df["games"] > 0)].copy()

# Define light and dark groups based on the 5-point scale
analysis_df["skin_group"] = pd.cut(
    analysis_df["skin_tone"],
    bins=[-0.01, 0.25, 0.5, 1.01],
    labels=["light", "medium", "dark"],
)

# Aggregate rate per game by group
summary = (
    analysis_df.groupby("skin_group", observed=True)
    .agg(
        dyads=("redCards", "size"),
        total_games=("games", "sum"),
        total_red=("redCards", "sum"),
    )
    .reset_index()
)
summary["red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson GLM with exposure for all skin_tone (continuous)
X = sm.add_constant(analysis_df[["skin_tone"]])
model = sm.GLM(analysis_df["redCards"], X, family=sm.families.Poisson(), offset=np.log(analysis_df["games"]))

# Try clustered SE by player, fallback to robust HC1
try:
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": analysis_df["playerShort"]})
    cov_type = "cluster_player"
except Exception:
    fit = model.fit(cov_type="HC1")
    cov_type = "HC1"

beta = fit.params["skin_tone"]
se = fit.bse["skin_tone"]
pval = fit.pvalues["skin_tone"]

# Rate ratio for dark (0.75) vs light (0.25) using continuous model
rate_ratio_cont = math.exp(beta * (0.75 - 0.25))

# Direct dark vs light comparison with binary indicator
binary_df = analysis_df.loc[analysis_df["skin_group"].isin(["light", "dark"])].copy()
binary_df["dark"] = (binary_df["skin_group"] == "dark").astype(int)
Xb = sm.add_constant(binary_df[["dark"]])
model_b = sm.GLM(binary_df["redCards"], Xb, family=sm.families.Poisson(), offset=np.log(binary_df["games"]))
try:
    fit_b = model_b.fit(cov_type="cluster", cov_kwds={"groups": binary_df["playerShort"]})
    cov_type_b = "cluster_player"
except Exception:
    fit_b = model_b.fit(cov_type="HC1")
    cov_type_b = "HC1"

beta_b = fit_b.params["dark"]
se_b = fit_b.bse["dark"]
pval_b = fit_b.pvalues["dark"]
rate_ratio_b = math.exp(beta_b)

result = {
    "n_rows": int(len(df)),
    "n_analysis": int(len(analysis_df)),
    "cov_type": cov_type,
    "cov_type_binary": cov_type_b,
    "summary": summary.to_dict(orient="records"),
    "beta_skin_tone": beta,
    "se_skin_tone": se,
    "pval_skin_tone": pval,
    "rate_ratio_dark_vs_light_cont": rate_ratio_cont,
    "beta_dark": beta_b,
    "se_dark": se_b,
    "pval_dark": pval_b,
    "rate_ratio_dark_vs_light_binary": rate_ratio_b,
}

print(json.dumps(result, indent=2))
