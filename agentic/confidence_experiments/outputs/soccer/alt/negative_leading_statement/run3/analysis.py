import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone (0=very light, 1=very dark)
df["skin"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Keep rows with skin tone and valid games
initial_rows = len(df)
df = df[df["games"].fillna(0) > 0]
df = df[~df["skin"].isna()]

# Define skin tone categories: light (<=0.25), dark (>=0.75)
df["skin_cat"] = pd.cut(df["skin"], bins=[-0.01, 0.25, 0.75, 1.01], labels=["light", "mid", "dark"])

# Group summaries for light vs dark
subset = df[df["skin_cat"].isin(["light", "dark"])].copy()

summary = subset.groupby("skin_cat", observed=True).agg(
    dyads=("redCards", "size"),
    players=("playerShort", "nunique"),
    total_games=("games", "sum"),
    total_red=("redCards", "sum"),
    mean_red_per_game=("redCards", lambda x: np.nan),
)
summary["mean_red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression: redCards ~ skin with exposure offset log(games)
# Use robust (HC0) SEs to mitigate mild misspecification
X = sm.add_constant(df["skin"])
model = sm.GLM(df["redCards"], X, family=sm.families.Poisson(), offset=np.log(df["games"]))
res = model.fit(cov_type="HC0")

# Alternative model with controls (league, position, age, height, weight, yellowCards)
# Compute age at 2012-2013 season midpoint (approx 2013-01-01)
# Birthday is dd.mm.yyyy
try:
    bday = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    df["age_2013"] = (pd.Timestamp("2013-01-01") - bday).dt.days / 365.25
except Exception:
    df["age_2013"] = np.nan

# Build design matrix with categorical controls
controls = [
    "skin",
    "age_2013",
    "height",
    "weight",
    "yellowCards",
]

# Drop rows with missing controls
df_ctrl = df.dropna(subset=controls + ["leagueCountry", "position", "games", "redCards"])

# One-hot encode categorical variables
X_ctrl = df_ctrl[controls].copy()
X_ctrl = pd.concat(
    [X_ctrl, pd.get_dummies(df_ctrl[["leagueCountry", "position"]], drop_first=True)],
    axis=1,
)
X_ctrl = sm.add_constant(X_ctrl)

model_ctrl = sm.GLM(
    df_ctrl["redCards"],
    X_ctrl,
    family=sm.families.Poisson(),
    offset=np.log(df_ctrl["games"]),
)
res_ctrl = model_ctrl.fit(cov_type="HC0")

# Extract effect for skin
coef = res.params["skin"]
se = res.bse["skin"]
pval = res.pvalues["skin"]
irr = float(np.exp(coef))

coef_c = res_ctrl.params.get("skin", np.nan)
se_c = res_ctrl.bse.get("skin", np.nan)
pval_c = res_ctrl.pvalues.get("skin", np.nan)
irr_c = float(np.exp(coef_c)) if pd.notna(coef_c) else np.nan

# Print results
print("Rows used:", len(df), "(from", initial_rows, ")")
print("Light vs dark summary (extremes):")
print(summary)
print("\nPoisson (redCards ~ skin + offset log(games))")
print(res.summary().tables[1])
print(f"IRR per 1.0 skin increase: {irr:.3f} (p={pval:.4g})")

print("\nPoisson with controls")
print("Skin coef (controls):", coef_c, "SE:", se_c, "p:", pval_c)
print(f"IRR per 1.0 skin increase (controls): {irr_c:.3f} (p={pval_c:.4g})")

# Also compute difference in red card rates between light and dark
if set(summary.index) == {"light", "dark"}:
    rate_light = summary.loc["light", "mean_red_per_game"]
    rate_dark = summary.loc["dark", "mean_red_per_game"]
    rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan
    print("\nRate (red cards per game) light:", rate_light)
    print("Rate (red cards per game) dark:", rate_dark)
    print("Rate ratio dark/light:", rate_ratio)
