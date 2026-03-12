import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"

df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric
for col in ["rater1", "rater2", "redCards", "games"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute mean skin tone from raters (allow if one missing)
df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

# Filter to rows with skin rating and positive games
analysis_df = df.loc[(df["skin_mean"].notna()) & (df["games"].notna()) & (df["games"] > 0) & (df["redCards"].notna())].copy()

# Define dark vs light based on 5-point scale normalized to 0..1
# Use conservative extremes to reflect "dark" vs "light" tones
analysis_df["skin_group"] = np.where(
    analysis_df["skin_mean"] >= 0.75, "dark",
    np.where(analysis_df["skin_mean"] <= 0.25, "light", "mid")
)

extremes_df = analysis_df[analysis_df["skin_group"].isin(["dark", "light"])].copy()

# Compute per-game red card rate
rate_summary = extremes_df.groupby("skin_group").apply(
    lambda g: pd.Series({
        "dyads": len(g),
        "total_games": g["games"].sum(),
        "total_reds": g["redCards"].sum(),
        "rate_per_game": g["redCards"].sum() / g["games"].sum() if g["games"].sum() > 0 else np.nan
    })
)

# Poisson regression with offset log(games)
# Model: redCards ~ dark_indicator + offset(log(games))
# Keep only extremes
extremes_df["dark"] = (extremes_df["skin_group"] == "dark").astype(int)

# Add constant
X = sm.add_constant(extremes_df["dark"])
y = extremes_df["redCards"].astype(float)

# Offset
offset = np.log(extremes_df["games"].astype(float))

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Also fit continuous skin tone model for robustness
X_cont = sm.add_constant(analysis_df["skin_mean"])
y_cont = analysis_df["redCards"].astype(float)
offset_cont = np.log(analysis_df["games"].astype(float))

poisson_cont = sm.GLM(y_cont, X_cont, family=sm.families.Poisson(), offset=offset_cont)
poisson_cont_res = poisson_cont.fit()

# Build results dictionary
results = {
    "n_total": int(len(analysis_df)),
    "n_extremes": int(len(extremes_df)),
    "rate_summary": rate_summary.to_dict(orient="index"),
    "poisson_extremes": {
        "coef_dark": float(poisson_res.params["dark"]),
        "se_dark": float(poisson_res.bse["dark"]),
        "p_dark": float(poisson_res.pvalues["dark"]),
        "irr_dark": float(np.exp(poisson_res.params["dark"])),
        "ci_dark": [
            float(np.exp(poisson_res.conf_int().loc["dark"][0])),
            float(np.exp(poisson_res.conf_int().loc["dark"][1]))
        ]
    },
    "poisson_cont": {
        "coef_skin": float(poisson_cont_res.params["skin_mean"]),
        "se_skin": float(poisson_cont_res.bse["skin_mean"]),
        "p_skin": float(poisson_cont_res.pvalues["skin_mean"]),
        "irr_per_unit": float(np.exp(poisson_cont_res.params["skin_mean"])),
        "ci_skin": [
            float(np.exp(poisson_cont_res.conf_int().loc["skin_mean"][0])),
            float(np.exp(poisson_cont_res.conf_int().loc["skin_mean"][1]))
        ]
    }
}

# Save intermediate results for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
