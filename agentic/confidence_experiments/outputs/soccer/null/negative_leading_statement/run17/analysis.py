import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Skin tone (0-1 scaled). Use mean of raters when available.
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

# Basic cleaning
df = df.replace([np.inf, -np.inf], np.nan)

# Keep rows with required fields
df_model = df.dropna(subset=["skin_tone", "games", "redCards"])
df_model = df_model[df_model["games"] > 0]

# Derive categorical comparisons
df_model["dark_half"] = (df_model["skin_tone"] >= 0.5).astype(int)
df_model["light_extreme"] = (df_model["skin_tone"] <= 0.25).astype(int)
df_model["dark_extreme"] = (df_model["skin_tone"] >= 0.75).astype(int)

# Summary rates
def rate_summary(mask, label):
    sub = df_model[mask]
    games = sub["games"].sum()
    reds = sub["redCards"].sum()
    rate = reds / games if games > 0 else np.nan
    return {
        "group": label,
        "rows": int(len(sub)),
        "games": float(games),
        "redCards": float(reds),
        "rate": float(rate),
    }

summaries = []
summaries.append(rate_summary(df_model["dark_half"] == 0, "light(<0.5)"))
summaries.append(rate_summary(df_model["dark_half"] == 1, "dark(>=0.5)"))
summaries.append(rate_summary(df_model["light_extreme"] == 1, "light_extreme(<=0.25)"))
summaries.append(rate_summary(df_model["dark_extreme"] == 1, "dark_extreme(>=0.75)"))

# Poisson GLM on dyads with offset for games
# Use cluster-robust SEs by player to reduce dependence
model = smf.glm(
    "redCards ~ skin_tone",
    data=df_model,
    family=sm.families.Poisson(),
    offset=np.log(df_model["games"]),
).fit(cov_type="cluster", cov_kwds={"groups": df_model["playerShort"]})

coef = model.params["skin_tone"]
se = model.bse["skin_tone"]
p = model.pvalues["skin_tone"]
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# Aggregated per player
agg = (
    df_model.groupby("playerShort")
    .agg(games=("games", "sum"), redCards=("redCards", "sum"), skin_tone=("skin_tone", "mean"))
    .reset_index()
)
agg = agg[agg["games"] > 0]

model_player = smf.glm(
    "redCards ~ skin_tone",
    data=agg,
    family=sm.families.Poisson(),
    offset=np.log(agg["games"]),
).fit()

coef_p = model_player.params["skin_tone"]
se_p = model_player.bse["skin_tone"]
p_p = model_player.pvalues["skin_tone"]
rr_p = np.exp(coef_p)
ci_low_p = np.exp(coef_p - 1.96 * se_p)
ci_high_p = np.exp(coef_p + 1.96 * se_p)

output = {
    "n_rows": int(len(df)),
    "n_model": int(len(df_model)),
    "summaries": summaries,
    "dyad_poisson": {
        "coef": float(coef),
        "se": float(se),
        "p": float(p),
        "rate_ratio": float(rr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    },
    "player_poisson": {
        "coef": float(coef_p),
        "se": float(se_p),
        "p": float(p_p),
        "rate_ratio": float(rr_p),
        "ci_low": float(ci_low_p),
        "ci_high": float(ci_high_p),
        "n_players": int(len(agg)),
    },
}

print(json.dumps(output, indent=2))
