import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone rating (0=very light, 1=very dark)
df["skin"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Basic cleaning
needed = ["redCards", "games", "skin", "playerShort", "position", "leagueCountry"]
clean = df.dropna(subset=needed).copy()

# Ensure numeric types
clean["redCards"] = pd.to_numeric(clean["redCards"], errors="coerce")
clean["games"] = pd.to_numeric(clean["games"], errors="coerce")
clean["skin"] = pd.to_numeric(clean["skin"], errors="coerce")
clean = clean.dropna(subset=["redCards", "games", "skin"]).copy()

# Avoid any nonpositive games (shouldn't exist)
clean = clean.loc[clean["games"] > 0].copy()

# Simple rate summaries by skin tone category
# Use extreme categories for clearer comparison
clean["skin_cat"] = np.where(clean["skin"] <= 0.25, "light",
                      np.where(clean["skin"] >= 0.75, "dark", "medium"))

rate_summary = (
    clean.groupby("skin_cat")
    .agg(
        dyads=("redCards", "size"),
        total_reds=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_skin=("skin", "mean")
    )
)
rate_summary["red_per_game"] = rate_summary["total_reds"] / rate_summary["total_games"]
rate_summary["red_per_100_games"] = rate_summary["red_per_game"] * 100

# Poisson regression with exposure offset (games)
# Model 1: skin only
model1 = smf.glm(
    formula="redCards ~ skin",
    data=clean,
    family=sm.families.Poisson(),
    offset=np.log(clean["games"])
)
res1 = model1.fit(cov_type="cluster", cov_kwds={"groups": clean["playerShort"]})

# Model 2: add position and league controls
model2 = smf.glm(
    formula="redCards ~ skin + C(position) + C(leagueCountry)",
    data=clean,
    family=sm.families.Poisson(),
    offset=np.log(clean["games"])
)
res2 = model2.fit(cov_type="cluster", cov_kwds={"groups": clean["playerShort"]})

# Extract key stats

def extract(res):
    coef = res.params["skin"]
    se = res.bse["skin"]
    pval = res.pvalues["skin"]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "rate_ratio": rr,
        "rr_ci95": [ci_low, ci_high],
    }

stats = {
    "n_dyads": int(clean.shape[0]),
    "n_players": int(clean["playerShort"].nunique()),
    "rate_summary": rate_summary.reset_index().to_dict(orient="records"),
    "model1": extract(res1),
    "model2": extract(res2),
}

with open("analysis_results.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
