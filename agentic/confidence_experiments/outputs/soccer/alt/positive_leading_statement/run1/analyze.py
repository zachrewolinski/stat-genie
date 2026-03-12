import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv("soccer.csv")

# Skin tone average from two raters
_df["skin_mean"] = _df[["rater1", "rater2"]].mean(axis=1)

# Define light and dark categories on 5-point normalized scale
light_mask = _df["skin_mean"] <= 0.25  # very light or light
Dark_mask = _df["skin_mean"] >= 0.75  # dark or very dark

# Keep only light or dark to answer the question directly
_df = _df[light_mask | Dark_mask].copy()
_df["dark"] = np.where(_df["skin_mean"] >= 0.75, 1, 0)

# Ensure positive exposure
_df = _df[_df["games"] > 0].copy()

# Summary stats
_df["red_per_game"] = _df["redCards"] / _df["games"]
summary = (
    _df.groupby("dark")
    .agg(
        dyads=("redCards", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_red_per_game=("red_per_game", "mean"),
    )
)

# Poisson regression with exposure offset
X = sm.add_constant(_df["dark"])
model = sm.GLM(
    _df["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(_df["games"]),
)
res = model.fit(cov_type="HC3")

beta = res.params["dark"]
se = res.bse["dark"]
pval = res.pvalues["dark"]
rate_ratio = float(np.exp(beta))
ci_low = float(np.exp(beta - 1.96 * se))
ci_high = float(np.exp(beta + 1.96 * se))

out = {
    "n_dyads": int(len(_df)),
    "summary": summary.reset_index().to_dict(orient="records"),
    "rate_ratio": rate_ratio,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "p_value": float(pval),
}

print(json.dumps(out, indent=2))
