import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path("soccer.csv")
INFO_PATH = Path("info.json")

# Load data
_df = pd.read_csv(DATA_PATH)

# Map columns for readability
# feature16: red cards count in dyad
# feature9: number of games in dyad (exposure)
# feature18/feature19: skin ratings (0-1), 5-point scale normalized

# Create average skin tone and drop rows with missing key vars
_df["skin_avg"] = _df[["feature18", "feature19"]].mean(axis=1)

# Ensure valid exposure (games)
_df = _df.dropna(subset=["skin_avg", "feature16", "feature9"])
_df = _df[_df["feature9"] > 0]

# Create any red card indicator
_df["red_any"] = (_df["feature16"] > 0).astype(int)

# Descriptive: rate per game by skin tone quantiles
_df["skin_bin"] = pd.qcut(_df["skin_avg"], q=4, labels=["lightest", "light", "dark", "darkest"])
rate_summary = (
    _df.groupby("skin_bin")
    .apply(lambda g: pd.Series({
        "red_cards": g["feature16"].sum(),
        "games": g["feature9"].sum(),
        "rate_per_game": g["feature16"].sum() / g["feature9"].sum(),
        "n": len(g)
    }))
)

# Poisson regression with log(games) offset
# Model: red cards ~ skin_avg, offset(log(games))
# Use robust SE due to overdispersion
_df["log_games"] = np.log(_df["feature9"])
poisson_model = smf.glm(
    formula="feature16 ~ skin_avg",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_games"],
).fit(cov_type="HC0")

# Logistic regression on any red card
logit_model = smf.logit("red_any ~ skin_avg", data=_df).fit(disp=False)

# Extract results
poisson_coef = poisson_model.params.get("skin_avg", np.nan)
poisson_se = poisson_model.bse.get("skin_avg", np.nan)
poisson_p = poisson_model.pvalues.get("skin_avg", np.nan)
poisson_rr = float(np.exp(poisson_coef)) if pd.notnull(poisson_coef) else np.nan

logit_coef = logit_model.params.get("skin_avg", np.nan)
logit_se = logit_model.bse.get("skin_avg", np.nan)
logit_p = logit_model.pvalues.get("skin_avg", np.nan)
logit_or = float(np.exp(logit_coef)) if pd.notnull(logit_coef) else np.nan

# Compute difference between darkest and lightest quartile rates for interpretation
rate_lightest = rate_summary.loc["lightest", "rate_per_game"]
rate_darkest = rate_summary.loc["darkest", "rate_per_game"]
rate_ratio_dark_vs_light = rate_darkest / rate_lightest if rate_lightest > 0 else np.nan

results = {
    "n_rows": int(len(_df)),
    "rate_summary": rate_summary.reset_index().to_dict(orient="records"),
    "poisson": {
        "coef": float(poisson_coef),
        "se": float(poisson_se),
        "p_value": float(poisson_p),
        "rate_ratio_per_1_unit_skin": float(poisson_rr),
    },
    "logit": {
        "coef": float(logit_coef),
        "se": float(logit_se),
        "p_value": float(logit_p),
        "odds_ratio_per_1_unit_skin": float(logit_or),
    },
    "rate_ratio_darkest_vs_lightest": float(rate_ratio_dark_vs_light),
}

print(json.dumps(results, indent=2))
