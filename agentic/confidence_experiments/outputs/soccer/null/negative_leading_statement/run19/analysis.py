import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute mean skin tone from two raters
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_mean=skin)

# Keep rows with skin tone and positive games
analysis_df = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Define light/dark groups using extreme bins of the 5-point scale
# rater values are normalized 0, 0.25, 0.5, 0.75, 1.0
analysis_df["light"] = analysis_df["skin_mean"] <= 0.25
analysis_df["dark"] = analysis_df["skin_mean"] >= 0.75

extremes = analysis_df[analysis_df["light"] | analysis_df["dark"]].copy()
extremes["dark_indicator"] = extremes["dark"].astype(int)

# Aggregate rates by group
agg = extremes.groupby("dark_indicator").agg(
    games_sum=("games", "sum"),
    red_sum=("redCards", "sum"),
    dyads=("redCards", "size"),
)
agg["rate_per_game"] = agg["red_sum"] / agg["games_sum"]

# Poisson regression with offset for exposure (games)
# Model: redCards ~ dark_indicator + offset(log(games))
endog = extremes["redCards"].astype(float)
exog = sm.add_constant(extremes["dark_indicator"].astype(float))
offset = np.log(extremes["games"].astype(float))

poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC0")

coef = poisson_res.params["dark_indicator"]
se = poisson_res.bse["dark_indicator"]
rr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))
p_value = float(poisson_res.pvalues["dark_indicator"])

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = float(poisson_res.pearson_chi2)
pearson_df = float(poisson_res.df_resid)
overdispersion = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

# Continuous skin tone model (per 0.25 step)
analysis_df["skin_step"] = analysis_df["skin_mean"] / 0.25
endog_all = analysis_df["redCards"].astype(float)
exog_all = sm.add_constant(analysis_df["skin_step"].astype(float))
offset_all = np.log(analysis_df["games"].astype(float))

poisson_model_all = sm.GLM(endog_all, exog_all, family=sm.families.Poisson(), offset=offset_all)
poisson_res_all = poisson_model_all.fit(cov_type="HC0")

coef_all = poisson_res_all.params["skin_step"]
se_all = poisson_res_all.bse["skin_step"]
rr_all = float(np.exp(coef_all))
ci_low_all = float(np.exp(coef_all - 1.96 * se_all))
ci_high_all = float(np.exp(coef_all + 1.96 * se_all))
p_value_all = float(poisson_res_all.pvalues["skin_step"])

pearson_chi2_all = float(poisson_res_all.pearson_chi2)
pearson_df_all = float(poisson_res_all.df_resid)
overdispersion_all = pearson_chi2_all / pearson_df_all if pearson_df_all > 0 else np.nan

# Rates by skin category (0, 0.25, 0.5, 0.75, 1.0)
analysis_df["skin_cat"] = (analysis_df["skin_mean"] * 4).round().clip(0, 4) / 4
cat_agg = analysis_df.groupby("skin_cat").agg(
    games_sum=("games", "sum"),
    red_sum=("redCards", "sum"),
    dyads=("redCards", "size"),
)
cat_agg["rate_per_game"] = cat_agg["red_sum"] / cat_agg["games_sum"]

results = {
    "n_rows_total": int(len(df)),
    "n_rows_with_skin": int(len(analysis_df)),
    "n_extreme_rows": int(len(extremes)),
    "group_stats": agg.reset_index().to_dict(orient="records"),
    "poisson_extremes": {
        "coef_dark": float(coef),
        "se_dark": float(se),
        "rate_ratio_dark_vs_light": rr,
        "ci95": [ci_low, ci_high],
        "p_value": p_value,
        "overdispersion": overdispersion,
    },
    "poisson_continuous": {
        "coef_per_skin_step": float(coef_all),
        "se_per_skin_step": float(se_all),
        "rate_ratio_per_skin_step": rr_all,
        "ci95": [ci_low_all, ci_high_all],
        "p_value": p_value_all,
        "overdispersion": overdispersion_all,
    },
    "rate_by_skin_category": cat_agg.reset_index().to_dict(orient="records"),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
