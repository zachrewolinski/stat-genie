import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone mean
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_mean=skin)

# Keep rows with skin ratings and games > 0
clean = df.loc[df["skin_mean"].notna() & (df["games"] > 0)].copy()

# Define light vs dark using clear extremes on the 5-point scale
# Light: <= 0.25 (very light/light), Dark: >= 0.75 (dark/very dark)
light = clean[clean["skin_mean"] <= 0.25].copy()
dark = clean[clean["skin_mean"] >= 0.75].copy()
subset = pd.concat([light, dark], axis=0).copy()
subset["dark"] = (subset["skin_mean"] >= 0.75).astype(int)

# Rates
subset["red_rate"] = subset["redCards"] / subset["games"]

# Group stats
group_stats = subset.groupby("dark").agg(
    n=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    mean_rate=("red_rate", "mean"),
)

# Poisson regression with offset (rate model)
# Controls: position and leagueCountry as categorical
model = smf.glm(
    formula="redCards ~ dark + C(position) + C(leagueCountry)",
    data=subset,
    family=sm.families.Poisson(),
    offset=np.log(subset["games"])  # exposure
).fit(cov_type="HC0")

# Extract effect of dark
coef_dark = model.params.get("dark", np.nan)
se_dark = model.bse.get("dark", np.nan)
rr_dark = np.exp(coef_dark) if pd.notna(coef_dark) else np.nan

# Save key results for later reading
results = {
    "group_stats": group_stats,
    "coef_dark": coef_dark,
    "se_dark": se_dark,
    "rr_dark": rr_dark,
    "pvalue_dark": model.pvalues.get("dark", np.nan),
}

# Print a concise summary for human inspection
print("Group stats (dark=0 light, dark=1 dark):")
print(group_stats)
print("\nPoisson GLM (rate) coef for dark:")
print({k: results[k] for k in ["coef_dark", "se_dark", "rr_dark", "pvalue_dark"]})
