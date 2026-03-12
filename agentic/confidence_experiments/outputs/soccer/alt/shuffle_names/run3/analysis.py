import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Identify columns based on metadata (names are shuffled)
# Skin tone ratings (normalized 0-1)
skin1_col = "rater1"
skin2_col = "nExp"
# Red card count (rare, 0-2) is in column 'yellowCards' per metadata
red_cards_col = "yellowCards"
# Games in dyad (exposure) is in column 'redCards' per metadata
exposure_col = "redCards"

# Basic cleaning
use_cols = [skin1_col, skin2_col, red_cards_col, exposure_col]
sub = df[use_cols].copy()

# Compute mean skin tone
sub["skin_mean"] = sub[[skin1_col, skin2_col]].mean(axis=1)

# Drop rows with missing values
sub = sub.dropna(subset=["skin_mean", red_cards_col, exposure_col])

# Ensure numeric and valid
sub = sub[(sub[exposure_col] > 0) & (sub[red_cards_col] >= 0)]

# Binary groups: light (<0.5), dark (>0.5), exclude neutral 0.5
sub["skin_group"] = np.where(sub["skin_mean"] > 0.5, "dark",
                              np.where(sub["skin_mean"] < 0.5, "light", "neutral"))

# Summary stats
summary = {
    "n_total": int(len(sub)),
    "n_dark": int((sub["skin_group"] == "dark").sum()),
    "n_light": int((sub["skin_group"] == "light").sum()),
    "n_neutral": int((sub["skin_group"] == "neutral").sum()),
}

# Red card rate per game by group
rate_by_group = (
    sub.groupby("skin_group")
       .apply(lambda g: pd.Series({
           "red_cards": g[red_cards_col].sum(),
           "games": g[exposure_col].sum(),
           "rate": g[red_cards_col].sum() / g[exposure_col].sum() if g[exposure_col].sum() > 0 else np.nan,
       }))
       .reset_index()
)

# Poisson regression with offset for games; predictor: skin_mean
# Use robust (HC0) standard errors
X = sm.add_constant(sub["skin_mean"])
offset = np.log(sub[exposure_col].astype(float))
model = sm.GLM(sub[red_cards_col], X, family=sm.families.Poisson(), offset=offset)
res = model.fit(cov_type="HC0")

coef = res.params["skin_mean"]
se = res.bse["skin_mean"]
pval = res.pvalues["skin_mean"]
rr = float(np.exp(coef))  # rate ratio per 1.0 increase in skin_mean

# Predicted rate for light (0.25) vs dark (0.75) skin_mean, holding exposure at 1 game
# (rates scale linearly with exposure in Poisson with offset)
light_val = 0.25
dark_val = 0.75
pred_light = float(res.predict([1, light_val], offset=np.log(1)))
pred_dark = float(res.predict([1, dark_val], offset=np.log(1)))

out = {
    "summary": summary,
    "rate_by_group": rate_by_group.to_dict(orient="records"),
    "poisson": {
        "coef_skin_mean": float(coef),
        "se_skin_mean": float(se),
        "p_value": float(pval),
        "rate_ratio_per_1_unit": rr,
        "pred_rate_light_0_25": pred_light,
        "pred_rate_dark_0_75": pred_dark,
    },
}

print(json.dumps(out, indent=2))
