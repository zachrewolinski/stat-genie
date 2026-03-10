import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load data
# Some columns are categorical; read all then convert numeric where needed.
df = pd.read_csv(DATA_PATH)

# Relevant columns
# feature9: games in dyad
# feature16: red cards
# feature18, feature19: skin ratings (0..1) by two raters

# Ensure numeric
for col in ["feature9", "feature16", "feature18", "feature19"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Average skin tone; drop rows without skin ratings
skin = df[["feature18", "feature19"]].mean(axis=1)

df = df.assign(skin=skin)

# Keep rows with games>0 and known skin
analysis_df = df[(df["feature9"] > 0) & df["skin"].notna() & df["feature16"].notna()].copy()

# Outcome and exposure
analysis_df["red_cards"] = analysis_df["feature16"].astype(float)
analysis_df["games"] = analysis_df["feature9"].astype(float)

# Poisson regression: red_cards ~ skin, offset log(games)
# Add intercept
X = sm.add_constant(analysis_df["skin"].astype(float))

glm = sm.GLM(
    analysis_df["red_cards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df["games"]) ,
)
res = glm.fit()

coef = res.params["skin"]
se = res.bse["skin"]
pval = res.pvalues["skin"]

irr = math.exp(coef)
# 95% CI for IRR
ci_low = math.exp(coef - 1.96 * se)
ci_high = math.exp(coef + 1.96 * se)

# Group comparison: light vs dark (use extreme categories)
light = analysis_df[analysis_df["skin"] <= 0.25]
dark = analysis_df[analysis_df["skin"] >= 0.75]

# Compute rates per game
light_rate = light["red_cards"].sum() / light["games"].sum() if light["games"].sum() > 0 else np.nan
dark_rate = dark["red_cards"].sum() / dark["games"].sum() if dark["games"].sum() > 0 else np.nan

# Poisson regression for group (dark vs light)
# Use only light/dark
ld = analysis_df[(analysis_df["skin"] <= 0.25) | (analysis_df["skin"] >= 0.75)].copy()
ld["dark"] = (ld["skin"] >= 0.75).astype(int)
X_ld = sm.add_constant(ld["dark"])

res_ld = sm.GLM(
    ld["red_cards"],
    X_ld,
    family=sm.families.Poisson(),
    offset=np.log(ld["games"]),
).fit()

coef_ld = res_ld.params["dark"]
se_ld = res_ld.bse["dark"]
pval_ld = res_ld.pvalues["dark"]

irr_ld = math.exp(coef_ld)
ci_low_ld = math.exp(coef_ld - 1.96 * se_ld)
ci_high_ld = math.exp(coef_ld + 1.96 * se_ld)

results = {
    "n_total": int(len(df)),
    "n_analysis": int(len(analysis_df)),
    "n_light": int(len(light)),
    "n_dark": int(len(dark)),
    "skin_coef": coef,
    "skin_se": se,
    "skin_pval": pval,
    "skin_irr": irr,
    "skin_irr_ci_low": ci_low,
    "skin_irr_ci_high": ci_high,
    "light_rate": light_rate,
    "dark_rate": dark_rate,
    "dark_vs_light_irr": irr_ld,
    "dark_vs_light_ci_low": ci_low_ld,
    "dark_vs_light_ci_high": ci_high_ld,
    "dark_vs_light_pval": pval_ld,
}

print(json.dumps(results, indent=2))
