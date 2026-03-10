import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.rates import test_poisson_2indep

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Map columns
RED = "feature16"
GAMES = "feature9"
SKIN1 = "feature18"
SKIN2 = "feature19"

# Average skin tone across raters
skin = df[[SKIN1, SKIN2]].mean(axis=1)

df = df.assign(skin_avg=skin)

# Drop rows without skin or games/red cards
analysis_df = df.dropna(subset=["skin_avg", RED, GAMES]).copy()

# Define light vs dark on normalized 0-1 scale (5-point)
# light: <=0.25 (very light / light), dark: >=0.75 (dark / very dark)
analysis_df["skin_group"] = np.where(
    analysis_df["skin_avg"] <= 0.25,
    "light",
    np.where(analysis_df["skin_avg"] >= 0.75, "dark", "mid"),
)

# Summaries for light and dark groups
ld_df = analysis_df[analysis_df["skin_group"].isin(["light", "dark"])].copy()

summary = (
    ld_df.groupby("skin_group")
    .agg(
        dyads=("skin_group", "size"),
        total_games=(GAMES, "sum"),
        total_red=(RED, "sum"),
        mean_red_per_game=(RED, lambda x: np.nan),
    )
)
summary["mean_red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson rate test between dark and light
count_dark = summary.loc["dark", "total_red"]
exposure_dark = summary.loc["dark", "total_games"]
count_light = summary.loc["light", "total_red"]
exposure_light = summary.loc["light", "total_games"]

rate_test = test_poisson_2indep(
    count1=count_dark,
    exposure1=exposure_dark,
    count2=count_light,
    exposure2=exposure_light,
    method="wald",
    alternative="larger",  # dark rate > light rate
)

rate_ratio = (count_dark / exposure_dark) / (count_light / exposure_light)

# Poisson regression with offset for games, continuous skin tone
reg_df = analysis_df.copy()
reg_df = reg_df[reg_df[GAMES] > 0]

X = sm.add_constant(reg_df["skin_avg"])
model = sm.GLM(reg_df[RED], X, family=sm.families.Poisson(), offset=np.log(reg_df[GAMES]))
res = model.fit(cov_type="HC0")

# Prepare results
results = {
    "n_rows": int(len(df)),
    "n_with_skin": int(len(analysis_df)),
    "group_summary": summary.reset_index().to_dict(orient="records"),
    "poisson_rate_test": {
        "count_dark": float(count_dark),
        "exposure_dark": float(exposure_dark),
        "count_light": float(count_light),
        "exposure_light": float(exposure_light),
        "rate_ratio": float(rate_ratio),
        "pvalue": float(rate_test.pvalue),
        "statistic": float(rate_test.statistic),
        "method": "wald",
        "alternative": "dark > light",
    },
    "poisson_regression": {
        "coef_skin": float(res.params["skin_avg"]),
        "se_skin": float(res.bse["skin_avg"]),
        "pvalue_skin": float(res.pvalues["skin_avg"]),
        "coef_intercept": float(res.params["const"]),
        "aic": float(res.aic),
    },
}

print(json.dumps(results, indent=2))
