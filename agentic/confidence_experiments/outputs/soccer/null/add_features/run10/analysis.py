import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute skin tone average if both raters available
# rater1/rater2 appear normalized to [0,1] in 0.25 increments
skin_avg = df[["rater1", "rater2"]].mean(axis=1)

# Identify light and dark categories using endpoints
# light: very light/light (<=0.25); dark: dark/very dark (>=0.75)
light_mask = skin_avg <= 0.25
dark_mask = skin_avg >= 0.75

# Keep rows with games > 0 and redCards not null
valid = df["games"].notna() & df["redCards"].notna() & df["games"].gt(0) & skin_avg.notna()

df_valid = df.loc[valid].copy()
df_valid["skin_avg"] = skin_avg[valid]

# Subset light and dark
light = df_valid.loc[light_mask[valid]].copy()
dark = df_valid.loc[dark_mask[valid]].copy()

# Basic counts
summary = {
    "total_rows": len(df),
    "valid_rows": len(df_valid),
    "light_rows": len(light),
    "dark_rows": len(dark),
}

# Rates per game
for label, d in [("light", light), ("dark", dark)]:
    total_reds = d["redCards"].sum()
    total_games = d["games"].sum()
    rate = total_reds / total_games if total_games > 0 else np.nan
    summary[f"{label}_total_reds"] = float(total_reds)
    summary[f"{label}_total_games"] = float(total_games)
    summary[f"{label}_reds_per_game"] = float(rate)

# Poisson regression with offset for games, binary indicator for dark vs light
# Use only light/dark rows
ld = pd.concat([light, dark], axis=0)
ld = ld.copy()
ld["dark"] = (ld["skin_avg"] >= 0.75).astype(int)

# Poisson GLM: redCards ~ dark + offset(log(games))
# Add small constant to games for safety
ld["log_games"] = np.log(ld["games"].astype(float))

X = sm.add_constant(ld[["dark"]])
model = sm.GLM(ld["redCards"], X, family=sm.families.Poisson(), offset=ld["log_games"])
res = model.fit()

# Extract rate ratio and p-value for dark
coef = res.params["dark"]
se = res.bse["dark"]
rate_ratio = float(np.exp(coef))
# 95% CI for rate ratio
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Wald p-value
p_value = float(res.pvalues["dark"])

# Also run a logistic regression for any red card occurrence (redCards>0) with offset? not typical.
# Instead, use binomial with games as exposure via weights and logit of per-game probability is not perfect.
# We'll do simple logistic on indicator with log(games) as covariate to adjust exposure.
ld["any_red"] = (ld["redCards"] > 0).astype(int)
X2 = sm.add_constant(ld[["dark", "games"]])
logit_model = sm.Logit(ld["any_red"], X2)
logit_res = logit_model.fit(disp=0)
logit_coef = logit_res.params["dark"]
logit_se = logit_res.bse["dark"]
logit_or = float(np.exp(logit_coef))
logit_ci_low = float(np.exp(logit_coef - 1.96 * logit_se))
logit_ci_high = float(np.exp(logit_coef + 1.96 * logit_se))
logit_p = float(logit_res.pvalues["dark"])

# Save outputs for review
print("SUMMARY")
for k, v in summary.items():
    print(f"{k}: {v}")
print("\nPOISSON")
print(f"rate_ratio: {rate_ratio}")
print(f"ci95: [{ci_low}, {ci_high}]")
print(f"p_value: {p_value}")
print("\nLOGIT")
print(f"odds_ratio: {logit_or}")
print(f"ci95: [{logit_ci_low}, {logit_ci_high}]")
print(f"p_value: {logit_p}")
