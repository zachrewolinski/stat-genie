import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"

df = pd.read_csv(path)

# Create skin tone average (0-1 scale)
skin_cols = ["feature18", "feature19"]
# If both present, average; if one missing, use available
skin = df[skin_cols].mean(axis=1)

df = df.copy()

df["skin_tone"] = skin

# Red cards count and games
red = df["feature16"]
games = df["feature9"]

# Basic counts
n_total = len(df)

# Filter valid rows: non-missing skin, positive games
valid = df["skin_tone"].notna() & games.notna() & (games > 0)

df_valid = df[valid].copy()

# Define light vs dark using quartiles (Q1 vs Q4)
q1 = df_valid["skin_tone"].quantile(0.25)
q3 = df_valid["skin_tone"].quantile(0.75)

light = df_valid[df_valid["skin_tone"] <= q1]
dark = df_valid[df_valid["skin_tone"] >= q3]

# Compute red card rates per game for groups
light_red = light["feature16"].sum()
light_games = light["feature9"].sum()

dark_red = dark["feature16"].sum()
dark_games = dark["feature9"].sum()

light_rate = light_red / light_games if light_games > 0 else np.nan
dark_rate = dark_red / dark_games if dark_games > 0 else np.nan

rate_ratio = dark_rate / light_rate if light_rate and not np.isnan(light_rate) else np.nan

# Poisson regression with offset for exposure (games)
# Use continuous skin tone
# Add small constant to avoid log(0) though games>0 filter should ensure

df_valid["log_games"] = np.log(df_valid["feature9"].astype(float))

# Model: red cards ~ skin_tone with offset
# Note: red cards are low counts; poisson should be fine for association

model = smf.glm(
    formula="feature16 ~ skin_tone",
    data=df_valid,
    family=sm.families.Poisson(),
    offset=df_valid["log_games"],
).fit()

coef = model.params.get("skin_tone", np.nan)
se = model.bse.get("skin_tone", np.nan)
pval = model.pvalues.get("skin_tone", np.nan)

# Translate coefficient to rate ratio for full scale difference (0 to 1)
rate_ratio_cont = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Robust SE via sandwich? Try if available
try:
    model_robust = model.get_robustcov_results(cov_type="HC1")
    coef_r = model_robust.params.get("skin_tone", np.nan)
    se_r = model_robust.bse.get("skin_tone", np.nan)
    pval_r = model_robust.pvalues.get("skin_tone", np.nan)
    rate_ratio_cont_r = float(np.exp(coef_r)) if np.isfinite(coef_r) else np.nan
except Exception:
    coef_r = se_r = pval_r = rate_ratio_cont_r = np.nan

# Save stats for later use
summary = {
    "n_total": int(n_total),
    "n_valid": int(len(df_valid)),
    "q1_skin": float(q1),
    "q3_skin": float(q3),
    "light_red": float(light_red),
    "light_games": float(light_games),
    "dark_red": float(dark_red),
    "dark_games": float(dark_games),
    "light_rate": float(light_rate),
    "dark_rate": float(dark_rate),
    "rate_ratio_q4_q1": float(rate_ratio),
    "poisson_coef": float(coef),
    "poisson_se": float(se),
    "poisson_pval": float(pval),
    "poisson_rate_ratio_0_1": float(rate_ratio_cont),
    "poisson_coef_robust": float(coef_r),
    "poisson_se_robust": float(se_r),
    "poisson_pval_robust": float(pval_r),
    "poisson_rate_ratio_0_1_robust": float(rate_ratio_cont_r),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
