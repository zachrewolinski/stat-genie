import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Create average skin tone from two raters
skin_cols = ["feature18", "feature19"]

df["skin_mean"] = df[skin_cols].mean(axis=1, skipna=True)

# Use only rows with skin ratings and valid games
base = df.dropna(subset=["skin_mean", "feature9", "feature16"]).copy()

# Guard against zero games (if any)
base = base[base["feature9"] > 0]

# Aggregate to player level (feature1 is short name)
agg = base.groupby("feature1", as_index=False).agg(
    games=("feature9", "sum"),
    red_cards=("feature16", "sum"),
    skin_mean=("skin_mean", "mean"),
)

# Define light and dark using scale endpoints (exclude middle) for clear contrast
# scale seems to be 0, 0.25, 0.5, 0.75, 1.0
# Light: <=0.25, Dark: >=0.75
agg["skin_group"] = pd.cut(
    agg["skin_mean"],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=["light", "mid", "dark"],
)

contrast = agg[agg["skin_group"].isin(["light", "dark"])].copy()

# Compute red card rates per game
contrast["red_rate"] = contrast["red_cards"] / contrast["games"]

summary = contrast.groupby("skin_group").agg(
    players=("feature1", "nunique"),
    games=("games", "sum"),
    red_cards=("red_cards", "sum"),
    mean_red_rate=("red_rate", "mean"),
).reset_index()

# Poisson regression with offset log(games)
# Use player-level totals for counts
contrast["skin_dark"] = (contrast["skin_group"] == "dark").astype(int)

poisson_model = sm.GLM(
    contrast["red_cards"],
    sm.add_constant(contrast[["skin_dark"]]),
    family=sm.families.Poisson(),
    offset=np.log(contrast["games"]),
).fit()

# Also compute a rate ratio and CI
coef = poisson_model.params["skin_dark"]
se = poisson_model.bse["skin_dark"]
rr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Simple two-sample test on rates (Welch t-test on player-level rates) for directionality
from scipy import stats

light_rates = contrast.loc[contrast["skin_group"] == "light", "red_rate"]
dark_rates = contrast.loc[contrast["skin_group"] == "dark", "red_rate"]

t_stat, p_val = stats.ttest_ind(dark_rates, light_rates, equal_var=False, nan_policy="omit")

result = {
    "summary": summary.to_dict(orient="records"),
    "poisson": {
        "coef": float(coef),
        "se": float(se),
        "rr": rr,
        "rr_ci_low": ci_low,
        "rr_ci_high": ci_high,
        "p_value": float(poisson_model.pvalues["skin_dark"]),
    },
    "ttest": {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "dark_mean_rate": float(dark_rates.mean()),
        "light_mean_rate": float(light_rates.mean()),
    },
    "counts": {
        "players_total": int(agg["feature1"].nunique()),
        "players_with_skin": int(agg["skin_mean"].notna().sum()),
        "players_light": int(contrast.loc[contrast["skin_group"] == "light", "feature1"].nunique()),
        "players_dark": int(contrast.loc[contrast["skin_group"] == "dark", "feature1"].nunique()),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
