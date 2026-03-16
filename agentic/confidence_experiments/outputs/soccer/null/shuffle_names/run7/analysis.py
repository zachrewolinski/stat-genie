import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Column mapping based on info.json descriptions
skin1_col = "rater1"   # rater1 skin rating normalized 0-1
skin2_col = "nExp"     # rater2 skin rating normalized 0-1
red_cards_col = "yellowCards"  # number of red cards
exposure_col = "redCards"      # number of games in dyad

# Clean data
for col in [skin1_col, skin2_col, red_cards_col, exposure_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Mean skin tone
skin_mean = df[[skin1_col, skin2_col]].mean(axis=1)

# Keep rows with exposure >0 and red cards >=0
mask = skin_mean.notna() & df[red_cards_col].notna() & df[exposure_col].notna() & (df[exposure_col] > 0) & (df[red_cards_col] >= 0)
use = df.loc[mask].copy()
use["skin_mean"] = skin_mean[mask]

# Create light vs dark groups using scale endpoints; exclude middle category 0.5
# Values are typically in {0, 0.25, 0.5, 0.75, 1}
use["skin_group"] = np.where(use["skin_mean"] >= 0.75, "dark", np.where(use["skin_mean"] <= 0.25, "light", "mid"))

# Group summary
summary = (
    use.groupby("skin_group")
    .agg(
        n=("skin_mean", "size"),
        red_cards=(red_cards_col, "sum"),
        games=(exposure_col, "sum"),
        mean_skin=("skin_mean", "mean"),
        mean_red_cards=(red_cards_col, "mean"),
        mean_games=(exposure_col, "mean"),
    )
)
summary["red_cards_per_game"] = summary["red_cards"] / summary["games"]

# Poisson regression with offset log(games)
# Use continuous skin_mean and then a binary model (dark vs light, exclude mid)

# Continuous model
X_cont = sm.add_constant(use["skin_mean"])
model_cont = sm.GLM(
    use[red_cards_col],
    X_cont,
    family=sm.families.Poisson(),
    offset=np.log(use[exposure_col])
)
res_cont = model_cont.fit()

# Binary model (dark vs light)
use_bin = use[use["skin_group"].isin(["dark", "light"])].copy()
use_bin["is_dark"] = (use_bin["skin_group"] == "dark").astype(int)
X_bin = sm.add_constant(use_bin["is_dark"])
model_bin = sm.GLM(
    use_bin[red_cards_col],
    X_bin,
    family=sm.families.Poisson(),
    offset=np.log(use_bin[exposure_col])
)
res_bin = model_bin.fit()

# Output key stats
print("Rows used:", len(use))
print("Rows used (dark/light only):", len(use_bin))
print("Summary by skin_group:\n", summary)
print("\nPoisson (continuous skin_mean) coef, IRR, p-value:")
coef_cont = res_cont.params["skin_mean"]
irr_cont = np.exp(coef_cont)
print({"coef": coef_cont, "IRR": irr_cont, "p": res_cont.pvalues["skin_mean"]})

print("\nPoisson (dark vs light) coef, IRR, p-value:")
coef_bin = res_bin.params["is_dark"]
irr_bin = np.exp(coef_bin)
print({"coef": coef_bin, "IRR": irr_bin, "p": res_bin.pvalues["is_dark"]})

# Also compute rate ratio directly
if "dark" in summary.index and "light" in summary.index:
    rr = summary.loc["dark", "red_cards_per_game"] / summary.loc["light", "red_cards_per_game"]
    print("\nRate ratio (dark/light) based on group rates:", rr)

# Save results for downstream use
res = {
    "summary": summary.to_dict(),
    "cont_coef": float(coef_cont),
    "cont_irr": float(irr_cont),
    "cont_p": float(res_cont.pvalues["skin_mean"]),
    "bin_coef": float(coef_bin),
    "bin_irr": float(irr_bin),
    "bin_p": float(res_bin.pvalues["is_dark"]),
}

import json
with open("analysis_results.json", "w") as f:
    json.dump(res, f, indent=2)
