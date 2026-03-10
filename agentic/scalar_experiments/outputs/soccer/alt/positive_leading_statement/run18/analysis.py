import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute mean skin tone from two raters
skin_mean = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_mean=skin_mean)

# Keep rows with skin ratings, games, redCards
analysis_df = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()
analysis_df = analysis_df[(analysis_df["games"] > 0)]

# Define light vs dark using extremes of 5-point scale (0, .25, .5, .75, 1)
light_mask = analysis_df["skin_mean"] <= 0.25
dark_mask = analysis_df["skin_mean"] >= 0.75
cat_df = analysis_df[light_mask | dark_mask].copy()
cat_df["dark"] = (cat_df["skin_mean"] >= 0.75).astype(int)

# Aggregate rates for descriptive stats
agg = cat_df.groupby("dark").agg(
    total_games=("games", "sum"),
    total_reds=("redCards", "sum"),
    n_rows=("redCards", "size"),
)
agg["reds_per_game"] = agg["total_reds"] / agg["total_games"]

# Poisson regression with offset for exposure (games)
X = sm.add_constant(cat_df["dark"])
offset = np.log(cat_df["games"].astype(float))
model = sm.GLM(cat_df["redCards"], X, family=sm.families.Poisson(), offset=offset)
result = model.fit(cov_type="HC1")

coef = result.params["dark"]
se = result.bse["dark"]
p_value = result.pvalues["dark"]
rate_ratio = float(np.exp(coef))

# Continuous model as sensitivity
Xc = sm.add_constant(analysis_df["skin_mean"])
offset_c = np.log(analysis_df["games"].astype(float))
model_c = sm.GLM(analysis_df["redCards"], Xc, family=sm.families.Poisson(), offset=offset_c)
result_c = model_c.fit(cov_type="HC1")
coef_c = result_c.params["skin_mean"]
p_value_c = result_c.pvalues["skin_mean"]
rate_ratio_c = float(np.exp(coef_c))

summary = {
    "n_total_rows": int(analysis_df.shape[0]),
    "n_cat_rows": int(cat_df.shape[0]),
    "agg_rates": {
        "light": {
            "total_games": float(agg.loc[0, "total_games"]) if 0 in agg.index else None,
            "total_reds": float(agg.loc[0, "total_reds"]) if 0 in agg.index else None,
            "reds_per_game": float(agg.loc[0, "reds_per_game"]) if 0 in agg.index else None,
            "n_rows": int(agg.loc[0, "n_rows"]) if 0 in agg.index else None,
        },
        "dark": {
            "total_games": float(agg.loc[1, "total_games"]) if 1 in agg.index else None,
            "total_reds": float(agg.loc[1, "total_reds"]) if 1 in agg.index else None,
            "reds_per_game": float(agg.loc[1, "reds_per_game"]) if 1 in agg.index else None,
            "n_rows": int(agg.loc[1, "n_rows"]) if 1 in agg.index else None,
        },
    },
    "poisson_dark_vs_light": {
        "coef": float(coef),
        "se": float(se),
        "p_value": float(p_value),
        "rate_ratio": rate_ratio,
    },
    "poisson_continuous": {
        "coef": float(coef_c),
        "p_value": float(p_value_c),
        "rate_ratio": rate_ratio_c,
    },
}

print(summary)
