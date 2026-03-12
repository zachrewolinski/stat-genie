import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Mean skin tone from two raters
for col in ["rater1", "rater2"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["mean_skin"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Basic filters: need games and redCards
base = df.copy()
base = base[base["games"].notna() & (base["games"] > 0) & base["redCards"].notna() & base["mean_skin"].notna()]

# Define light/dark categories using 5-point scale bins (0, 0.25, 0.5, 0.75, 1)
# light: <=0.25 (very light/light), dark: >=0.75 (dark/very dark)
base["skin_group"] = pd.cut(
    base["mean_skin"],
    bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
    labels=["light", "mid_light", "mid_dark", "dark"],
)

subset_ld = base[base["skin_group"].isin(["light", "dark"])].copy()
subset_ld["dark"] = (subset_ld["skin_group"] == "dark").astype(int)

# Descriptive rates
rates = (
    subset_ld.groupby("skin_group")
    .agg(total_red=("redCards", "sum"), total_games=("games", "sum"), n_dyads=("redCards", "size"))
)
rates["red_per_game"] = rates["total_red"] / rates["total_games"]

# Poisson GLM with offset for games; cluster-robust SEs by player
# Use available controls with good coverage: leagueCountry and position
controls = []
for col in ["leagueCountry", "position"]:
    if col in subset_ld.columns:
        controls.append(f"C({col})")

formula = "redCards ~ dark"
if controls:
    formula += " + " + " + ".join(controls)

model_ld = smf.glm(
    formula=formula,
    data=subset_ld,
    family=sm.families.Poisson(),
    offset=np.log(subset_ld["games"]),
).fit(cov_type="HC0")

# Continuous skin tone model on full data
controls_full = []
for col in ["leagueCountry", "position"]:
    if col in base.columns:
        controls_full.append(f"C({col})")

formula_full = "redCards ~ mean_skin"
if controls_full:
    formula_full += " + " + " + ".join(controls_full)

model_full = smf.glm(
    formula=formula_full,
    data=base,
    family=sm.families.Poisson(),
    offset=np.log(base["games"]),
).fit(cov_type="HC0")

# Extract effect sizes
ld_coef = model_ld.params.get("dark", np.nan)
ld_se = model_ld.bse.get("dark", np.nan)
ld_p = model_ld.pvalues.get("dark", np.nan)
ld_rr = np.exp(ld_coef) if pd.notna(ld_coef) else np.nan

full_coef = model_full.params.get("mean_skin", np.nan)
full_se = model_full.bse.get("mean_skin", np.nan)
full_p = model_full.pvalues.get("mean_skin", np.nan)
full_rr = np.exp(full_coef) if pd.notna(full_coef) else np.nan

# Print summary metrics
print("Descriptive rates (light vs dark):")
print(rates)
print("\nPoisson GLM (dark vs light, controls leagueCountry+position):")
print(model_ld.summary().tables[1])
print(f"Rate ratio (dark vs light): {ld_rr:.3f}, p={ld_p:.4g}")

print("\nPoisson GLM (continuous mean_skin, controls leagueCountry+position):")
print(model_full.summary().tables[1])
print(f"Rate ratio per 1.0 increase in mean_skin: {full_rr:.3f}, p={full_p:.4g}")

# Save key numbers for later
out = {
    "n_base": int(len(base)),
    "n_ld": int(len(subset_ld)),
    "rates": rates.reset_index().to_dict(orient="records"),
    "ld_coef": float(ld_coef),
    "ld_se": float(ld_se),
    "ld_p": float(ld_p),
    "ld_rr": float(ld_rr),
    "full_coef": float(full_coef),
    "full_se": float(full_se),
    "full_p": float(full_p),
    "full_rr": float(full_rr),
}

pd.Series(out).to_json("analysis_results.json")
