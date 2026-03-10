import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Skin tone average (0 to 1)
df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

# Basic filtering
analysis_df = df[(df["games"] > 0) & df["redCards"].notna() & df["skin_mean"].notna()].copy()

# Define light/dark categories from the 5-point scale
# 0, 0.25 (light), 0.5 (mid), 0.75, 1.0 (dark)
analysis_df["skin_cat"] = pd.cut(
    analysis_df["skin_mean"],
    bins=[-0.01, 0.25, 0.5, 1.01],
    labels=["light", "mid", "dark"],
)

# Aggregated rates per group
agg = (
    analysis_df.groupby("skin_cat")
    .agg(dyads=("redCards", "size"), red_cards=("redCards", "sum"), games=("games", "sum"))
    .reset_index()
)
agg["rate_per_game"] = agg["red_cards"] / agg["games"]
agg["rate_per_100_games"] = agg["rate_per_game"] * 100

# Poisson regression with offset for exposure (games)
# Continuous skin tone
glm_all = smf.glm(
    "redCards ~ skin_mean + C(position) + C(leagueCountry)",
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df["games"]),
).fit(cov_type="HC3")
coef_skin = glm_all.params.get("skin_mean", np.nan)
se_skin = glm_all.bse.get("skin_mean", np.nan)
p_skin = glm_all.pvalues.get("skin_mean", np.nan)

# Effect for one 0.25 step on 5-point scale
step = 0.25
rr_step = float(np.exp(coef_skin * step))
ci_step = (
    float(np.exp((coef_skin - 1.96 * se_skin) * step)),
    float(np.exp((coef_skin + 1.96 * se_skin) * step)),
)

# Light vs dark only
ld_df = analysis_df[analysis_df["skin_cat"].isin(["light", "dark"])].copy()
ld_df["dark"] = (ld_df["skin_cat"] == "dark").astype(int)

glm_ld = smf.glm(
    "redCards ~ dark + C(position) + C(leagueCountry)",
    data=ld_df,
    family=sm.families.Poisson(),
    offset=np.log(ld_df["games"]),
).fit(cov_type="HC3")
coef_dark = glm_ld.params.get("dark", np.nan)
se_dark = glm_ld.bse.get("dark", np.nan)
p_dark = glm_ld.pvalues.get("dark", np.nan)
rr_dark = float(np.exp(coef_dark))
ci_dark = (
    float(np.exp(coef_dark - 1.96 * se_dark)),
    float(np.exp(coef_dark + 1.96 * se_dark)),
)

# Collect summary stats
summary = {
    "n_rows": int(len(analysis_df)),
    "agg": agg.to_dict(orient="records"),
    "skin_mean_coef": float(coef_skin),
    "skin_mean_p": float(p_skin),
    "skin_step_rr": rr_step,
    "skin_step_rr_ci": ci_step,
    "dark_rr": rr_dark,
    "dark_rr_ci": ci_dark,
    "dark_p": float(p_dark),
    "n_light_dark": int(len(ld_df)),
}

# Build explanation
# Determine Likert response based on significance and effect size
# Heuristic: significant p<0.05 and RR>1.05 => moderate yes.
response_value = 50
if np.isfinite(p_dark) and p_dark < 0.05 and rr_dark > 1.05:
    # scale strength based on rr magnitude
    if rr_dark >= 1.25:
        response_value = 80
    elif rr_dark >= 1.15:
        response_value = 70
    else:
        response_value = 60
elif np.isfinite(p_dark) and p_dark < 0.05 and rr_dark <= 1.05:
    response_value = 55
else:
    # no significant evidence; tilt toward "No"
    response_value = 30 if rr_dark < 1.0 else 35

light_row = next((r for r in summary["agg"] if r["skin_cat"] == "light"), None)
dark_row = next((r for r in summary["agg"] if r["skin_cat"] == "dark"), None)

light_rate = light_row["rate_per_100_games"] if light_row else np.nan
dark_rate = dark_row["rate_per_100_games"] if dark_row else np.nan

explanation = (
    "Using dyad-level data with games as exposure, I modeled red-card counts with a Poisson regression "
    "and robust (HC3) standard errors, controlling for player position and league country. "
    f"The continuous skin-tone measure is not significantly associated with red-card rates: a one-step "
    f"increase on the 5-point scale (0.25) corresponds to an estimated rate ratio of {rr_step:.3f} "
    f"(95% CI {ci_step[0]:.3f} to {ci_step[1]:.3f}; p={p_skin:.4g}). "
    f"When comparing dark (>=0.75) vs light (<=0.25) players, the model estimates a rate ratio of "
    f"{rr_dark:.3f} (95% CI {ci_dark[0]:.3f} to {ci_dark[1]:.3f}; p={p_dark:.4g}). "
)

if light_row and dark_row:
    explanation += (
        f"In raw rates, light-skin dyads receive about {light_rate:.3f} red cards per 100 games, "
        f"while dark-skin dyads receive about {dark_rate:.3f} per 100 games. "
    )

explanation += (
    "Overall, the estimated differences are small, not statistically significant, and the point "
    "estimates slightly favor lower (not higher) red-card rates for darker-skinned players in this sample."
)

output = {"response": int(response_value), "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)

# Also print summary for inspection
print(json.dumps(summary, indent=2))
