import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Skin tone: average of two raters (normalized 0-1)
skin = df[["feature18", "feature19"]].mean(axis=1, skipna=True)

# Keep rows with skin ratings and games > 0
mask = skin.notna() & (df["feature9"] > 0)

df = df.loc[mask].copy()
df["skin_mean"] = skin[mask]

# Define light vs dark: exclude neutral (0.5)
df = df[df["skin_mean"] != 0.5].copy()
df["dark"] = (df["skin_mean"] > 0.5).astype(int)

# Basic summaries
summary = df.groupby("dark").agg(
    dyads=("feature16", "size"),
    total_games=("feature9", "sum"),
    total_reds=("feature16", "sum"),
)
summary["reds_per_game"] = summary["total_reds"] / summary["total_games"]
summary["reds_per_100_games"] = summary["reds_per_game"] * 100

# Poisson regression with log(games) offset
endog = df["feature16"].astype(float)
exog = sm.add_constant(df["dark"].astype(float))
offset = np.log(df["feature9"].astype(float))

poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC3")

coef = poisson_res.params["dark"]
se = poisson_res.bse["dark"]

irr = math.exp(coef)
ci_low = math.exp(coef - 1.96 * se)
ci_high = math.exp(coef + 1.96 * se)

# Negative binomial as robustness check
nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset)
nb_res = nb_model.fit(cov_type="HC3")

coef_nb = nb_res.params["dark"]
se_nb = nb_res.bse["dark"]
irr_nb = math.exp(coef_nb)
ci_low_nb = math.exp(coef_nb - 1.96 * se_nb)
ci_high_nb = math.exp(coef_nb + 1.96 * se_nb)

results = {
    "summary": summary.reset_index().to_dict(orient="records"),
    "poisson": {
        "coef": coef,
        "se": se,
        "p_value": float(poisson_res.pvalues["dark"]),
        "irr": irr,
        "ci_low": ci_low,
        "ci_high": ci_high,
    },
    "neg_bin": {
        "coef": coef_nb,
        "se": se_nb,
        "p_value": float(nb_res.pvalues["dark"]),
        "irr": irr_nb,
        "ci_low": ci_low_nb,
        "ci_high": ci_high_nb,
    },
    "n": int(len(df)),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
