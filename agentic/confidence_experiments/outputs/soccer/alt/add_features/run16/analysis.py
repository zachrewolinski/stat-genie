import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Average skin tone from two raters
for col in ["rater1", "rater2"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["skin_avg"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with known skin tone and valid games
analysis_df = df.loc[df["skin_avg"].notna() & df["games"].notna()].copy()
analysis_df = analysis_df[analysis_df["games"] > 0]

# Define light and dark groups using endpoints of 5-point scale
# Values are normalized: 0, 0.25, 0.5, 0.75, 1.0
analysis_df["skin_group"] = np.where(
    analysis_df["skin_avg"] <= 0.25,
    "light",
    np.where(analysis_df["skin_avg"] >= 0.75, "dark", "mid")
)

ld_df = analysis_df[analysis_df["skin_group"].isin(["light", "dark"])].copy()

# Outcome variables
ld_df["redCards"] = pd.to_numeric(ld_df["redCards"], errors="coerce").fillna(0)
ld_df["games"] = pd.to_numeric(ld_df["games"], errors="coerce")
ld_df = ld_df[ld_df["games"] > 0]

ld_df["red_rate"] = ld_df["redCards"] / ld_df["games"]
ld_df["any_red"] = (ld_df["redCards"] > 0).astype(int)
ld_df["dark"] = (ld_df["skin_group"] == "dark").astype(int)

summary = {
    "n_rows": int(len(df)),
    "n_with_skin": int(analysis_df.shape[0]),
    "n_light": int((ld_df["skin_group"] == "light").sum()),
    "n_dark": int((ld_df["skin_group"] == "dark").sum()),
    "mean_red_rate_light": float(ld_df.loc[ld_df["skin_group"] == "light", "red_rate"].mean()),
    "mean_red_rate_dark": float(ld_df.loc[ld_df["skin_group"] == "dark", "red_rate"].mean()),
    "any_red_light": float(ld_df.loc[ld_df["skin_group"] == "light", "any_red"].mean()),
    "any_red_dark": float(ld_df.loc[ld_df["skin_group"] == "dark", "any_red"].mean()),
}

# Poisson regression with offset for games (rate model)
X = sm.add_constant(ld_df["dark"])
poisson_model = sm.GLM(ld_df["redCards"], X, family=sm.families.Poisson(), offset=np.log(ld_df["games"]))
poisson_res = poisson_model.fit(cov_type="HC0")

summary["poisson_coef_dark"] = float(poisson_res.params["dark"])
summary["poisson_pvalue_dark"] = float(poisson_res.pvalues["dark"])
summary["poisson_irr_dark"] = float(np.exp(poisson_res.params["dark"]))
summary["poisson_ci_low"] = float(np.exp(poisson_res.conf_int().loc["dark", 0]))
summary["poisson_ci_high"] = float(np.exp(poisson_res.conf_int().loc["dark", 1]))

# Logistic regression for any red card, controlling for games as covariate
X_logit = sm.add_constant(ld_df[["dark", "games"]])
logit_model = sm.Logit(ld_df["any_red"], X_logit)
logit_res = logit_model.fit(disp=False)

summary["logit_coef_dark"] = float(logit_res.params["dark"])
summary["logit_pvalue_dark"] = float(logit_res.pvalues["dark"])
summary["logit_or_dark"] = float(np.exp(logit_res.params["dark"]))
summary["logit_ci_low"] = float(np.exp(logit_res.conf_int().loc["dark", 0]))
summary["logit_ci_high"] = float(np.exp(logit_res.conf_int().loc["dark", 1]))

# Save summary for inspection
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
