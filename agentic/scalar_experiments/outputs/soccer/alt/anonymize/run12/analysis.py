import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute average skin tone across raters
skin_avg = df[["feature18", "feature19"]].mean(axis=1)
df = df.assign(skin_avg=skin_avg)

# Basic filters
# Use only rows with skin ratings and positive games
mask_valid = df["skin_avg"].notna() & df["feature9"].notna() & (df["feature9"] > 0)
df_valid = df.loc[mask_valid].copy()

# Define light and dark categories (extremes)
light_mask = df_valid["skin_avg"] <= 0.25
dark_mask = df_valid["skin_avg"] >= 0.75

# Adjust if extremes too small (fallback to median split)
if light_mask.sum() < 1000 or dark_mask.sum() < 1000:
    median = df_valid["skin_avg"].median()
    light_mask = df_valid["skin_avg"] <= median
    dark_mask = df_valid["skin_avg"] > median

# Prepare grouped summaries
subset = df_valid.loc[light_mask | dark_mask].copy()
subset["dark"] = (subset["skin_avg"] >= (0.75 if (light_mask.sum() >= 1000 and dark_mask.sum() >= 1000) else subset["skin_avg"].median())).astype(int)

# If we used median split, ensure dark label matches mask
if (light_mask.sum() < 1000 or dark_mask.sum() < 1000):
    subset["dark"] = (subset["skin_avg"] > df_valid["skin_avg"].median()).astype(int)

# Aggregate rates
summary = subset.groupby("dark").agg(
    rows=("skin_avg", "size"),
    total_games=("feature9", "sum"),
    total_red=("feature16", "sum")
).reset_index()
summary["rate_per_game"] = summary["total_red"] / summary["total_games"]
summary["rate_per_100_games"] = summary["rate_per_game"] * 100

# Poisson regression with offset log(games)
X = sm.add_constant(subset["dark"])
model_pois = sm.GLM(subset["feature16"], X, family=sm.families.Poisson(), offset=np.log(subset["feature9"]))
res_pois = model_pois.fit()

# Negative binomial regression (robust to overdispersion)
model_nb = sm.GLM(subset["feature16"], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(subset["feature9"]))
res_nb = model_nb.fit()

# Continuous skin tone effect
X_cont = sm.add_constant(df_valid["skin_avg"])
model_pois_cont = sm.GLM(df_valid["feature16"], X_cont, family=sm.families.Poisson(), offset=np.log(df_valid["feature9"]))
res_pois_cont = model_pois_cont.fit()

# Logistic regression: probability of any red card
subset["any_red"] = (subset["feature16"] > 0).astype(int)
X_logit = sm.add_constant(subset[["dark", "feature9"]])
try:
    model_logit = sm.Logit(subset["any_red"], X_logit)
    res_logit = model_logit.fit(disp=False)
except Exception:
    res_logit = None

results = {
    "n_rows": int(len(df)),
    "n_valid": int(len(df_valid)),
    "group_summary": summary.to_dict(orient="records"),
    "poisson": {
        "coef_dark": float(res_pois.params["dark"]),
        "p_dark": float(res_pois.pvalues["dark"]),
        "rr_dark": float(np.exp(res_pois.params["dark"])),
        "ci_low": float(np.exp(res_pois.conf_int().loc["dark", 0])),
        "ci_high": float(np.exp(res_pois.conf_int().loc["dark", 1])),
    },
    "neg_binom": {
        "coef_dark": float(res_nb.params["dark"]),
        "p_dark": float(res_nb.pvalues["dark"]),
        "rr_dark": float(np.exp(res_nb.params["dark"])),
        "ci_low": float(np.exp(res_nb.conf_int().loc["dark", 0])),
        "ci_high": float(np.exp(res_nb.conf_int().loc["dark", 1])),
    },
    "poisson_cont": {
        "coef_skin": float(res_pois_cont.params["skin_avg"]),
        "p_skin": float(res_pois_cont.pvalues["skin_avg"]),
        "rr_per_unit": float(np.exp(res_pois_cont.params["skin_avg"])),
        "ci_low": float(np.exp(res_pois_cont.conf_int().loc["skin_avg", 0])),
        "ci_high": float(np.exp(res_pois_cont.conf_int().loc["skin_avg", 1])),
    },
    "logit_any_red": None,
}

if res_logit is not None:
    results["logit_any_red"] = {
        "coef_dark": float(res_logit.params["dark"]),
        "p_dark": float(res_logit.pvalues["dark"]),
        "odds_ratio": float(np.exp(res_logit.params["dark"])),
        "ci_low": float(np.exp(res_logit.conf_int().loc["dark", 0])),
        "ci_high": float(np.exp(res_logit.conf_int().loc["dark", 1])),
    }

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
