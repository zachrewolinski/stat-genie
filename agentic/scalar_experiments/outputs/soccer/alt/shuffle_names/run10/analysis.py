import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Columns based on info.json descriptions (names are shuffled)
# player id
player_col = "photoID"  # short player name
# skin ratings
r1_col = "rater1"
r2_col = "nExp"
# red cards count per dyad
red_col = "yellowCards"
# games/exposure per dyad
games_col = "redCards"

# Keep relevant columns
cols = [player_col, r1_col, r2_col, red_col, games_col]
sub = df[cols].copy()

# Coerce to numeric where needed
for c in [r1_col, r2_col, red_col, games_col]:
    sub[c] = pd.to_numeric(sub[c], errors='coerce')

# Compute skin tone mean per row
sub["skin_mean"] = sub[[r1_col, r2_col]].mean(axis=1)

# Drop rows with missing
sub = sub.dropna(subset=["skin_mean", red_col, games_col, player_col])

# Aggregate to player level
agg = (
    sub.groupby(player_col)
       .agg(
            skin_mean=("skin_mean", "mean"),
            red_cards=(red_col, "sum"),
            games=(games_col, "sum"),
            n_rows=(red_col, "size")
       )
       .reset_index()
)

# Remove players with zero games or missing
agg = agg[(agg["games"] > 0) & agg["red_cards"].notna() & agg["skin_mean"].notna()]

# Create dark/light groups
agg["dark"] = (agg["skin_mean"] >= 0.5).astype(int)

# Summary statistics
summary = agg.groupby("dark").apply(
    lambda g: pd.Series({
        "players": g.shape[0],
        "red_cards": g["red_cards"].sum(),
        "games": g["games"].sum(),
        "rate_per_game": g["red_cards"].sum() / g["games"].sum() if g["games"].sum() > 0 else np.nan,
        "avg_skin": g["skin_mean"].mean(),
    })
)

# Poisson regression on player-level counts with exposure offset
X = sm.add_constant(agg[["skin_mean"]])
model = sm.GLM(agg["red_cards"], X, family=sm.families.Poisson(), offset=np.log(agg["games"]))
res = model.fit(cov_type="HC0")

# Extract coefficient and IRR
coef = res.params["skin_mean"]
se = res.bse["skin_mean"]
pval = res.pvalues["skin_mean"]
irr = np.exp(coef)
ci_low, ci_high = np.exp(res.conf_int().loc["skin_mean"].values)

# Rate ratio for dark vs light groups
rate_dark = summary.loc[1, "rate_per_game"] if 1 in summary.index else np.nan
rate_light = summary.loc[0, "rate_per_game"] if 0 in summary.index else np.nan
rate_ratio = (rate_dark / rate_light) if (rate_dark is not None and rate_light is not None and rate_light > 0) else np.nan

print("Players:", agg.shape[0])
print("Dark group summary:\n", summary)
print("Poisson GLM (player-level, offset log(games))")
print(res.summary())
print(f"skin_mean coef={coef:.4f} se={se:.4f} p={pval:.4g} IRR={irr:.4f} CI=[{ci_low:.4f},{ci_high:.4f}]")
print(f"Rate light={rate_light:.6f}, dark={rate_dark:.6f}, ratio={rate_ratio:.4f}")
